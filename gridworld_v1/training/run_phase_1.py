import papermill as pm

# defaults
# initialization
learning_rate = 0.001
initial_epsilon = 1
epsilon_decay = 0.9995
final_epsilon = 0.05
discount_factor = 0.99

episodes = 10000
max_steps_multiplier = 4
target_update_freq = 500 # for target CNN, in steps
visibility = 4

goal_reward = 10
step_penalty = -0.5

num_special_regions = 1
special_region_rewards = [-5.0]

experience_capacity = 8000
batch_size = 64

grid_sizes = [5, 9] # training on different sizes for better generalizability, these grid sizes are arbitrary

num_filters_first_layer = 16
final_conv_filters = num_filters_first_layer * 2
target_spatial_size = 1

changes = None # will be overriden by papermill if running headlessly, and if not I'll get from input

# end copy from phase_1.ipynb

# update this! 
experiments = [
    {
        "changes": "test1", 
        "episodes": 1000, # just testing if it works
    }, 
    {
        "changes": "lower visibility", 
        "visibility": 3, 
        "episodes": 10_000
    }, 
    {
        "changes": "lower discount factor", 
        "discount_factor": 0.9, 
        "episodes": 10_000
    }
]

for run_id, experiment in enumerate(experiments, start=1): 
    inputs = {
        "changes": experiment["changes"], 
        "learning_rate": experiment.get("learning_rate", learning_rate),
        "epsilon_decay": experiment.get("epsilon_decay", epsilon_decay),
        "final_epsilon": experiment.get("final_epsilon", final_epsilon),
        "discount_factor": experiment.get("discount_factor", discount_factor),
        "episodes": experiment.get("episodes", episodes),
        "max_steps_multiplier": experiment.get("max_steps_multiplier", max_steps_multiplier),
        "target_update_freq": experiment.get("target_update_freq", target_update_freq),
        "visibility": experiment.get("visibility", visibility),
        "goal_reward": experiment.get("goal_reward", goal_reward),
        "step_penalty": experiment.get("step_penalty", step_penalty),
        "num_special_regions": experiment.get("num_special_regions", num_special_regions),
        "special_region_rewards": experiment.get("special_region_rewards", special_region_rewards),
        "experience_capacity": experiment.get("experience_capacity", experience_capacity),
        "batch_size": experiment.get("batch_size", batch_size),
        "grid_sizes": experiment.get("grid_sizes", grid_sizes),
        "num_filters_first_layer": experiment.get("num_filters_first_layer", num_filters_first_layer),
        "final_conv_filters": experiment.get("num_filters_first_layer", num_filters_first_layer) * 2,
        "target_spatial_size": experiment.get("target_spatial_size", target_spatial_size),

        "notes": "Ran in the background, may add notes later if this run was important/a checkpoint"
    }

    pm.execute_notebook(
        'phase_1.ipynb', 
        'phase_1_output.ipynb', 
        parameters=inputs, # will configure this in .ipynb later
        log_output=False
    )

    print(f"Finished execution for {run_id}")

print("Finished execution for all scheduled runs")