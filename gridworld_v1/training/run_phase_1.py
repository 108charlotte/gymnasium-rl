import papermill as pm

# defaults
mode = "train" # default
model_folder_path = None
# initialization
learning_rate = 0.001
initial_epsilon = 1
epsilon_decay = 0.999 # reduced
final_epsilon = 0.05
discount_factor = 0.99

episodes = 10000
max_steps_multiplier = 5
target_update_freq = 500 # for target CNN, in steps
visibility = 4

goal_reward = 10
step_penalty = -0.1

num_special_regions = 1
special_region_rewards = [-5.0]
num_directions = 16 # this is the default anyways
reward_shaping = True

experience_capacity = 8000
batch_size = 64

spawn_widths = [5, 10, 15, 20] # matters for target placement

num_filters_first_layer = 16
final_conv_filters = num_filters_first_layer * 2
target_spatial_size = 3
compass_penalty_multiplier = 0.05

changes = None # will be overriden by papermill if running headlessly, and if not I'll get from input
notes = None
# end copy from phase_1.ipynb


experiments = [
    {
        "changes": "testing without special regions", 
        "num_special_regions": 0
    }, 
    {
        "changes": "testing with lower discount factor", 
        "discount_factor": 0.9
    }
]

for run_id, experiment in enumerate(experiments, start=1): 
    inputs = {
        "changes": experiment["changes"], 
        "mode": experiment.get("mode", mode), 
        "compass_penalty_multiplier": experiment.get("compass_penalty_multiplier", compass_penalty_multiplier), 
        "model_folder_path": experiment.get("model_folder_path", model_folder_path), 
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
        "spawn_widths": experiment.get("spawn_widths", spawn_widths),
        "num_filters_first_layer": experiment.get("num_filters_first_layer", num_filters_first_layer),
        "final_conv_filters": experiment.get("num_filters_first_layer", num_filters_first_layer) * 2,
        "target_spatial_size": experiment.get("target_spatial_size", target_spatial_size),

        "notes": "Ran in the background, may add notes later if this run was important/a checkpoint"
    }

    pm.execute_notebook(
        'phase_1.ipynb', 
        'phase_1_output.ipynb', # don't super care about this so fine with it being overriden each run
        parameters=inputs, # will configure this in .ipynb later
        log_output=True
    )

    print(f"Finished execution for {run_id}")

print("Finished execution for all scheduled runs")