import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --- Supply Chain Parameters ---
NUM_SUPPLIERS =3
NUM_PRODUCTS = 1
WAREHOUSE_CAPACITY = 100  # Example
DISTRIBUTION_CENTER_CAPACITY = 25 #Example

HOLDING_COST_PER_UNIT = 0.01  # Reduced holding cost
ORDERING_COST_PER_ORDER = 2  # Reduced ordering cost
SHORTAGE_COST_PER_UNIT = 0.5  # Reduced shortage cost
TRANSPORT_COST_PER_UNIT_DISTANCE = 0.005  # Reduced transport cost

# --- Demand Parameters ---
AVG_DEMAND_PER_PRODUCT = 15  # Example
DEMAND_STD_DEV = 2  # Example

# --- Lead Times (in days) ---
SUPPLIER_LEAD_TIMES = [random.randint(1, 3) for _ in range(NUM_SUPPLIERS)]  # Random lead times between 2 and 6 days
TRANSPORT_LEAD_TIME_WAREHOUSE_TO_DC = 1  # Days
PRICING_REQUEST_APPROVAL_PROBABILITY = 0.8 #probability of getting  the request accepted

class SupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super(SupplyChainEnv, self).__init__()

        self.render_mode = render_mode
        # Define the state space (inventory levels at warehouse and DC for each product)
        # We'll represent this as a flattened array.
        # We need to account for each location (warehouse, DC)
        # and each product.  Plus, let's add cash on hand.

        # Calculate the total size of the inventory state.  We will use unbounded for now.
        inventory_state_size = (1 + 1) * NUM_PRODUCTS  # Warehouse + DC for each product

        # State space: [warehouse_inv_prod_1, ..., warehouse_inv_prod_10, dc_inv_prod_1, ..., dc_inv_prod_10, cash_on_hand]
        self.observation_space = spaces.Box(
            low=0,  # Inventory cannot be negative. Cash cannot be negative
            high=WAREHOUSE_CAPACITY,  # Inventory has a capacity limit
            shape=(inventory_state_size + 1,),  # +1 for cash on hand.  Important: shape needs to be a tuple!
            dtype=np.float32
        )

        # Action Space: Order Quantities (from each supplier for each product) and Pricing request

        # Ordering actions
        # Actions: For each product and supplier, specify the order quantity.
        # and one price increase/decrease for one product
        self.order_action_size = NUM_SUPPLIERS * NUM_PRODUCTS

        # Pricing actions
        self.price_action_size = NUM_PRODUCTS # price change request for each product

        # Combine both the ordering and pricing
        self.action_space = spaces.Box(
            low=0,  # Order quantities cannot be negative
            high=WAREHOUSE_CAPACITY,  #Maximum is warehouse capacity. The price changes can be in range -1 to 1
            shape=(self.order_action_size + self.price_action_size ,),
            dtype=np.float32
        )

        # Internal State Variables
        self.warehouse_inventory = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        self.distribution_center_inventory = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        self.supplier_backorders = np.zeros((NUM_SUPPLIERS, NUM_PRODUCTS), dtype=np.float32)  # Amount owed by suppliers
        self.demand_backorders = np.zeros(NUM_PRODUCTS, dtype=np.float32)   #Demand that is yet to be fulfilled
        self.cash_on_hand = 20000  # Increased initial cash
        self.current_day = 0

        # Incoming shipments (supplier, product, quantity, arrival_time)
        self.incoming_shipments = []
        self.shipment_dc_retailer = [] # warehouse to DC

        # Price of products
        self.product_prices = np.ones(NUM_PRODUCTS, dtype=np.float32) * 10 #inital price for the product
        self.price_change_amount = 0.1  # Amount by which the price can change(10% increase or decrease)

    def _get_obs(self):
        # Combine warehouse and distribution center inventory into a single state vector.
        inventory_state = np.concatenate([self.warehouse_inventory, self.distribution_center_inventory])
        return np.concatenate([inventory_state, [self.cash_on_hand]]).astype(np.float32)

    def _get_info(self):
        return {
            "warehouse_inventory": self.warehouse_inventory.tolist(),
            "distribution_center_inventory": self.distribution_center_inventory.tolist(),
            "cash_on_hand": self.cash_on_hand,
            "product_prices" : self.product_prices.tolist()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.warehouse_inventory = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        self.distribution_center_inventory = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        self.supplier_backorders = np.zeros((NUM_SUPPLIERS, NUM_PRODUCTS), dtype=np.float32)
        self.demand_backorders = np.zeros(NUM_PRODUCTS, dtype=np.float32)
        self.cash_on_hand = 30000  # Increased initial cash
        self.current_day = 0
        self.incoming_shipments = []
        self.shipment_dc_retailer = []
        self.product_prices = np.ones(NUM_PRODUCTS, dtype=np.float32) * 10

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Split the action into ordering and pricing.
        order_actions = action[:self.order_action_size]
        price_actions = action[self.order_action_size:]

        # Reshape the ordering actions into a (NUM_SUPPLIERS, NUM_PRODUCTS) array
        order_quantities = order_actions.reshape((NUM_SUPPLIERS, NUM_PRODUCTS))

        # 1. Place Orders with Suppliers
        self._place_orders(order_quantities)

        # 2. Process Incoming Shipments
        self._process_incoming_shipments()

        # 3. Process Shipments From warehouse to DC
        self._process_shipment_warehouse_to_dc()

        # 4. Fulfill Demand (from Distribution Center)
        demand, revenue = self._fulfill_demand()

        # 5. Update Prices (after fulfilling demand)
        self._update_prices(price_actions)

        # 6. Calculate Costs
        holding_cost, ordering_cost, shortage_cost, transport_cost = self._calculate_costs(order_quantities)

        # 7. Calculate Reward
        reward = (
            revenue
            - (holding_cost + ordering_cost + shortage_cost + transport_cost)
            - 0.5 * np.sum(self.distribution_center_inventory)  # Penalize excessive inventory
            - 0.3 * np.sum(self.demand_backorders)  # Penalize backorders
            + 1.0 * revenue  # Stronger incentive for revenue
            + 0.1 * np.sum(order_quantities)  # Incentivize placing orders
        )

        # 8. Update Cash
        self.cash_on_hand += reward

        # 9. Increment Day
        self.current_day += 1

        # Display the day's summary to the owner
        print(f"\n--- Day {self.current_day} Summary ---")
        print(f"Revenue: {revenue:.2f}")
        print(f"Costs Breakdown:")
        print(f"  - Holding Cost: {holding_cost:.2f}")
        print(f"  - Ordering Cost: {ordering_cost:.2f}")
        print(f"  - Shortage Cost: {shortage_cost:.2f}")
        print(f"  - Transport Cost: {transport_cost:.2f}")
        print(f"Cash Balance: {self.cash_on_hand:.2f}")
        print(f"Warehouse Inventory: {self.warehouse_inventory.tolist()}")
        print(f"Distribution Center Inventory: {self.distribution_center_inventory.tolist()}")
        print(f"Demand Backorders: {self.demand_backorders.tolist()}")

        # Ask the owner if they want to proceed with ordering goods
        proceed = input("Do you want to proceed with ordering goods for the next day? (yes/no): ").strip().lower()
        if proceed == "yes":
            # Use RL model to suggest order quantities
            suggested_action, _ = model.predict(self._get_obs())
            suggested_order_quantities = suggested_action[:self.order_action_size].reshape((NUM_SUPPLIERS, NUM_PRODUCTS))

            # Allow the owner to input custom order quantities
            print("\nEnter custom order quantities for each supplier and product (suggested quantities are shown):")
            for supplier in range(NUM_SUPPLIERS):
                for product in range(NUM_PRODUCTS):
                    try:
                        suggested_quantity = int(suggested_order_quantities[supplier, product])
                        quantity = int(input(f"Supplier {supplier + 1}, Product {product + 1} (Suggested: {suggested_quantity}): "))
                        order_quantities[supplier, product] = max(0, quantity)  # Ensure non-negative quantities
                    except ValueError:
                        print("Invalid input. Defaulting to suggested quantity.")
                        order_quantities[supplier, product] = suggested_quantity

            # Place the custom orders
            self._place_orders(order_quantities)
        else:
            print("Owner decided not to order goods. Using RL-based fallback ordering.")
            # Use RL-based fallback ordering
            fallback_action, _ = model.predict(self._get_obs())  # Predict action using the RL model
            fallback_order_quantities = fallback_action[:self.order_action_size].reshape((NUM_SUPPLIERS, NUM_PRODUCTS))
            self._place_orders(fallback_order_quantities)

        # Check the termination
        terminated = False
        if self.cash_on_hand <= 0:
            print("Simulation terminated: Cash depleted.")
            raise SystemExit("Program stopped: Cash balance is zero or negative.")

        if self.current_day >= 60:  # Extended simulation duration
            terminated = True
            raise SystemExit("60 din pure huye.")

        observation = self._get_obs()
        info = self._get_info()

        print(f"Day {self.current_day}: Revenue={revenue:.2f}, Costs={holding_cost + ordering_cost + shortage_cost + transport_cost:.2f}, Cash={self.cash_on_hand:.2f}")

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _place_orders(self, order_quantities):
        """Places orders with suppliers.

        Args:
            order_quantities (np.ndarray): A (NUM_SUPPLIERS, NUM_PRODUCTS) array
                                           representing the quantity of each product
                                           to order from each supplier.
        """
        total_ordering_cost = 0
        for supplier in range(NUM_SUPPLIERS):
            for product in range(NUM_PRODUCTS):
                quantity = order_quantities[supplier, product]

                # Automatically place orders if inventory is critically low
                if self.distribution_center_inventory[product] < AVG_DEMAND_PER_PRODUCT:
                    quantity = max(quantity, AVG_DEMAND_PER_PRODUCT - self.distribution_center_inventory[product])

                # Ensure we have enough cash to place the order
                product_cost = 10  # Assuming product price = 10
                if self.cash_on_hand >= quantity * product_cost:
                    self.cash_on_hand -= quantity * product_cost
                    total_ordering_cost += quantity * product_cost
                    self.incoming_shipments.append(
                        (supplier, product, quantity, self.current_day + SUPPLIER_LEAD_TIMES[supplier])
                    )
                    print(f"Ordered {quantity} units of Product {product + 1} from Supplier {supplier + 1} at {product_cost} per unit.")

        print(f"Total Amount Spent on Ordering Goods: {total_ordering_cost:.2f}")

    def _process_incoming_shipments(self):
        """Processes shipments that have arrived at the warehouse."""
        arrived_shipments = [s for s in self.incoming_shipments if s[3] <= self.current_day]
        for supplier, product, quantity, _ in arrived_shipments:
            #Fulfill the backorder first
            if self.supplier_backorders[supplier, product] > 0:
              delivered = min(quantity, self.supplier_backorders[supplier, product])
              self.warehouse_inventory[product] += delivered
              self.supplier_backorders[supplier, product] -= delivered
              quantity -= delivered

            #Now check for warehouse capacity
            available_capacity = WAREHOUSE_CAPACITY - self.warehouse_inventory[product]
            received = min(quantity, available_capacity) #How much can be received

            #Update Inventory
            self.warehouse_inventory[product] += received

            #Update the backorders
            if quantity > available_capacity:
              self.supplier_backorders[supplier, product] += quantity - available_capacity

        #Remove
        self.incoming_shipments = [s for s in self.incoming_shipments if s[3] > self.current_day]

        #Now send to the distributions center
        for product in range(NUM_PRODUCTS):
            #Ship as much as possible
            available_capacity = DISTRIBUTION_CENTER_CAPACITY - self.distribution_center_inventory[product]
            transport = min(self.warehouse_inventory[product], available_capacity)
            self.warehouse_inventory[product] -= transport #Remove form the warehouse
            self.shipment_dc_retailer.append((product, transport, self.current_day+TRANSPORT_LEAD_TIME_WAREHOUSE_TO_DC)) #create the incoming shipment

    def _process_shipment_warehouse_to_dc(self):
        """Processes shipments from warehouse to DC."""
        arrived_shipments = [s for s in self.shipment_dc_retailer if s[2] <= self.current_day]
        for product, quantity, _ in arrived_shipments:
            self.distribution_center_inventory[product] += quantity

        #Remove
        self.shipment_dc_retailer = [s for s in self.shipment_dc_retailer if s[2] > self.current_day]

    def _forecast_demand(self, product):
        # Simpler demand forecasting using random sampling
        demand = np.random.randint(AVG_DEMAND_PER_PRODUCT - DEMAND_STD_DEV, AVG_DEMAND_PER_PRODUCT + DEMAND_STD_DEV)
        return max(0, demand)  # Ensure non-negative demand

    def _fulfill_demand(self):
        total_demand = 0
        total_revenue = 0
        # Sort products by price (descending) to prioritize high-revenue products
        sorted_products = np.argsort(-self.product_prices)
        for product in sorted_products:
            demand = self._forecast_demand(product)

            # Fulfill backorders first
            if self.demand_backorders[product] > 0:
                delivered = min(demand, self.demand_backorders[product])
                self.distribution_center_inventory[product] -= delivered
                demand -= delivered
                self.demand_backorders[product] -= delivered

            # Fulfill remaining demand
            fulfilled = min(demand, self.distribution_center_inventory[product])
            self.distribution_center_inventory[product] -= fulfilled
            demand -= fulfilled

            # Add unmet demand to backorders
            if demand > 0:
                self.demand_backorders[product] += demand

            # Calculate revenue
            total_revenue += fulfilled * self.product_prices[product]

        return total_demand, total_revenue

    def _update_prices(self, price_actions):
        """Updates the product prices based on the provided actions and inventory levels."""
        for product in range(NUM_PRODUCTS):
            # Adjust prices based on inventory and demand
            if self.distribution_center_inventory[product] < AVG_DEMAND_PER_PRODUCT:
                # Increase price if inventory is low
                self.product_prices[product] = min(self.product_prices[product] * 1.1, 100)
            elif self.distribution_center_inventory[product] > 2 * AVG_DEMAND_PER_PRODUCT:
                # Decrease price if inventory is high
                self.product_prices[product] = max(self.product_prices[product] * 0.9, 1)

            # Apply price actions (if approved)
            random_number = random.random()
            if 0 < self.product_prices[product] + (price_actions[product] * self.price_change_amount) < 100 and \
                    random_number <= PRICING_REQUEST_APPROVAL_PROBABILITY:
                self.product_prices[product] += (price_actions[product] * self.price_change_amount)

    def _calculate_costs(self, order_quantities):
        """Calculates the total cost for the current time step."""
        holding_cost = np.sum(self.warehouse_inventory * HOLDING_COST_PER_UNIT) + \
                      np.sum(self.distribution_center_inventory * HOLDING_COST_PER_UNIT)
        ordering_cost = np.sum(order_quantities > 0) * ORDERING_COST_PER_ORDER  # Simplified: cost per order
        shortage_cost = np.sum(self.demand_backorders * SHORTAGE_COST_PER_UNIT)
        transport_cost = np.sum(self.shipment_dc_retailer) * TRANSPORT_COST_PER_UNIT_DISTANCE

        return holding_cost, ordering_cost, shortage_cost, transport_cost

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        plt.figure(figsize=(10, 6))
        plt.bar(range(NUM_PRODUCTS), self.warehouse_inventory, label="Warehouse Inventory")
        plt.bar(range(NUM_PRODUCTS), self.distribution_center_inventory, label="DC Inventory", alpha=0.7)
        plt.title(f"Day {self.current_day} - Cash: {self.cash_on_hand:.2f}")
        plt.xlabel("Products")
        plt.ylabel("Inventory Levels")
        plt.legend()
        plt.show()

    def close(self):
        pass

# --- Example Usage ---
env = make_vec_env(lambda: SupplyChainEnv(render_mode=None), n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Train the RL agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)

# Save the trained model
model.save("supply_chain_agent")

# Load the trained model
model = PPO.load("supply_chain_agent")