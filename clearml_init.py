from clearml import Task

# Initialize a new task
task = Task.init(project_name='GuardianAI', task_name='my_first_experiment')

# Your ML code goes here
# ClearML will automatically log:
# - Code execution
# - Python packages and versions
# - Model parameters
# - Training metrics
# - Model artifacts

# Example of manually logging parameters
parameters = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
}
task.connect(parameters)

# Example of logging metrics
task.logger.report_scalar(
    title='Performance Metrics',
    series='accuracy',
    value=0.85,
    iteration=1
) 