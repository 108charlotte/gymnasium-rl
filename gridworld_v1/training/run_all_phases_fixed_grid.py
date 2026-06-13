import papermill as pm

# default
mode = "train" # default
model_folder_path = None # for loading a model if in test
# initialization
learning_rate = 0.001
initial_epsilon = 1.0
epsilon_decay = 0.9995
final_epsilon = 0.05
discount_factor = 0.99

episodes = 10000
max_steps = 50
target_update_freq = 500 # for target CNN, in steps

goal_reward = 10
step_penalty = -0.5

num_special_regions = 1
special_region_rewards = [-5.0]

experience_capacity = 8000
batch_size = 64

world_size = 6 # arbitrary choice, what matters is that I'm only using one
num_filters_first_layer = 16
final_conv_filters = num_filters_first_layer * 2
target_spatial_size = 3

changes = None # will be overriden by papermill if running headlessly, and if not I'll get from input
notes = None
# end copy from fixed_size_all_phases.ipynb


experiments = [
    {
        "changes": "Re-running with more info: Target spatial size the same as world size",
        "target_spatial_size": world_size, 
        "episodes": 30_000,
    }, 
    {
        "changes": "Re-running with more info: Very high special region penalty",
        "special_region_penalty": [-10.0], # I tried this before, but now in isolation
        "episodes": 30_000,
    }, 
    {
        "changes": "Re-running with more info: Both target spatial size the same as world size and very high special region penalty combined", 
        "target_spatial_size": world_size, 
        "special_region_penalty": [-10.0],
        "episodes": 30_000, 
    }
]


for run_id, experiment in enumerate(experiments, start=1): 
    inputs = {
        "changes": experiment["changes"], 
        "mode": experiment.get("mode", mode), 
        "model_folder_path": experiment.get("model_folder_path", model_folder_path), 
        "learning_rate": experiment.get("learning_rate", learning_rate),
        "epsilon_decay": experiment.get("epsilon_decay", epsilon_decay),
        "final_epsilon": experiment.get("final_epsilon", final_epsilon),
        "discount_factor": experiment.get("discount_factor", discount_factor),
        "episodes": experiment.get("episodes", episodes),
        "max_steps": experiment.get("max_steps", max_steps),
        "target_update_freq": experiment.get("target_update_freq", target_update_freq),
        "goal_reward": experiment.get("goal_reward", goal_reward),
        "step_penalty": experiment.get("step_penalty", step_penalty),
        "num_special_regions": experiment.get("num_special_regions", num_special_regions),
        "special_region_rewards": experiment.get("special_region_rewards", special_region_rewards),
        "experience_capacity": experiment.get("experience_capacity", experience_capacity),
        "batch_size": experiment.get("batch_size", batch_size),
        "world_size": experiment.get("world_size", world_size),
        "num_filters_first_layer": experiment.get("num_filters_first_layer", num_filters_first_layer),
        "final_conv_filters": experiment.get("num_filters_first_layer", num_filters_first_layer) * 2,
        "target_spatial_size": experiment.get("target_spatial_size", target_spatial_size),

        "notes": "Ran in the background, may add notes later if this run was important/a checkpoint"
    }

    pm.execute_notebook(
        'fixed_size_all_phases.ipynb', 
        'fixed_size_all_phases_output.ipynb', # don't super care about this so fine with it being overriden each run
        parameters=inputs, 
        log_output=True
    )

    print(f"Finished execution for {run_id}")

print("Finished execution for all scheduled runs")
