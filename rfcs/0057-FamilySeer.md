- Feature Name: (FamilySeer: A new search method for Auto-scheduler)
- Start Date: (2021-01-07)
- RFC PR: [apache/tvm-rfcs#57](https://github.com/apache/tvm-rfcs/pull/57)
- GitHub Issue: [apache/tvm#9875](https://github.com/apache/tvm/pull/9875)

# Summary
[summary]: #summary

We propose FamilySeer, a new search method that optimizes search efficiency and quality of the Auto-scheduler. We introduce several features:

- FamilySeer exploits the subgraph similarity to form a collection of subgraph families and constructs cost models at subgraph family basis to improve cost model accuracy.
- We enable subgraphs within each family to share the search results within each tuning iteration, avoiding costly code measurements on real hardware and thus accelerating the search process to converge to optimal results.
- We also make some general optimizations like enabling parallel measurement on single node with multiple GPUs and training the cost model on GPU.

# Motivation
[motivation]: #motivation

Auto-scheduler (Ansor) uses code sketch and optimization rules to generate a large search space. The search space defined by Ansor has shown great opportunities and therefore the search quality and the search efficiency are determined by how we search the space.

Ansor utilizes improved cost model and task scheduler to help explore the search space. The cost model analyzes and finds high-performance code transformations in the search space and the task scheduler allocates the time budget to different computation graphs. However, we find serval drawbacks to this approach:

The accuracy of the cost model determines the search quality, but Ansor uses monolithic cost model to predict different computation graphs (subgraphs), resulting in an accuracy loss during tuning.

The task scheduler allocates most of the time budget to subgraphs with most improving potential (i.e., those with the highest latency). This approach works well at the beginning of the autotuning. However, as the potential subgraph gradually reaches its peak performance with adequate time budget, other subgraphs have little time budget to reach its peak performance.

The search process will at the end take a dozen of hours. This motivates us to find better way to explore the search space.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

We integrate our search method into Auto-Scheduler. Therefore, users only need to change some of the parameters to enable our search method.

We use the code below in [Auto-scheduling a Neural Network for NVIDIA GPU](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/tune_network_cuda.html#begin-tuning) as an example:

```python
#...

# load all task and into tuner
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)

#define tuning option for tuner
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,  # change this to 20000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)

# start tuning
#tuner.tune(tune_option) #add new parameter to tune function 
tuner.tune(tune_option,search_policy="sketch.xgb.family_op")

```

The `tuner` loads the `tune_option` into the `tune` function. There are several parameters in the `tune` function (Refer to class [Taskscheduler](https://tvm.apache.org/docs/reference/api/python/auto_scheduler.html?highlight=taskscheduler#tvm.auto_scheduler.TaskScheduler)). Users can enable our method by changing the `search_policy` parameter to `sketch.xgb.family_<family_algorithm>`. We currently provide two family algorithms as an option: `op` refers to classifying subgraphs based on the core operation, and `hash` refers to classifying subgraphs based on operation sequence. We recommend using `op` to achieve better performance.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

Our search method consists of three steps:

1. Identifying similar subgraphs
```python
def make_family_group(
  tasks,
  search_policy,
):
  """identify each subgraphs and group them into subgraph family.
  """
  if search_policy == "default":
      search_policy = "sketch.xgb"

  if isinstance(search_policy, str):
      policy = search_policy.split(".")
      if len(policy) == 2:
          return {}
      elif len(policy) == 3:
          _, _, model_group = policy
          _, class_type = model_group.split("_")
      else:
          raise ValueError("Invalid search policy: " + search_policy)
      
  family_group = {}
  if class_type == "op":
      for idx, task in enumerate(tasks):
          task_layers = task.desc.split('_')
          if task_layers[1] not in family_group:
              family_group[task_layers[1]] = []
              family_group[task_layers[1]].append(idx)
          else:
              family_group[task_layers[1]].append(idx)

  elif class_type == "hash":
      for idx, task in enumerate(tasks):
          first = task.workload_key.find("[\"") + 2
          end = task.workload_key.find("\",")
          task_hash = task.workload_key[first:end]
          if task_hash not in family_group:
              family_group[task_hash] = []
              family_group[task_hash].append(idx)
          else:
              family_group[task_hash].append(idx)

  elif class_type == "ind":
      for idx, task in enumerate(tasks):
          if task.workload_key not in family_group:
              family_group[task.workload_key] = []
              family_group[task.workload_key].append(idx)
          else:
              family_group[task.workload_key].append(idx)
      
  if family_group is not None:
      for key, value in family_group.items():
          print("family group :", key, "---", value)

  return family_group
```

We use static analyzing to classify the subgraphs(tasks) based on their attributes. 

2. Constructing family cost model
```python
elif "family" in model_group:
  # generate cost model for each family
  cost_model_pool = []
  for _,group in family_group.items():
    if model_type == "xgb":
      cost_model = XGBModel(
          num_warmup_sample=len(group) * num_measures_per_round,
          model_file=load_model_file,
          adapative_training=adapative_training,
      )
      if load_model_file and os.path.isfile(load_model_file):
        logger.info("TaskScheduler: Load pretrained model...")
        cost_model.load(load_model_file)
      elif load_log_file:
        logger.info("TaskScheduler: Reload measured states and train the model...")
        cost_model.update_from_file(load_log_file)
    elif model_type == "random":
        cost_model = RandomModel()
    else:
        raise ValueError("Invalid search policy: " + search_policy)
    cost_model_pool.append(cost_model)
  
  #bind each subgraph(task) with its family cost model
  search_policies = []
  for task_idx,task in enumerate(tasks):
    for group_idx,group in enumerate(family_group.values()):
        if task_idx in group:
          search_policies.append(
            SketchPolicy(
                task,
                cost_model_pool[group_idx],
                params=search_policy_params,
                verbose=verbose,
                init_search_callbacks=init_search_callbacks,
            )
          )

```

After identifying similar subgraphs, We return `family_group` (a list of subgraph families) and build a cost model for each subgraph family.

3.Foresee tuning

```python
def _tune_family_task(self, task_idx_groups,skip_measures_per_round):
  """Tune the select family task for one round.
  """
  for task_idx in task_idx_groups:
    # Run pre-tune callbacks
    for callback in self.callbacks:
      callback.pre_tune(self, task_idx)

    measure_inputs, measure_results = self.search_policies[task_idx].continue_search_one_round(
      skip_measures_per_round, self.measurer
    )

    ……
```

The foresee tuning takes `task_idx_groups` (A list of subgraph families) and `skip_measures_per_round` as inputs and tunes all the subgraphs inside the list. 

# Drawbacks
[drawbacks]: #drawbacks

When searching on a larger search space (such as larger batch size), FamilySeer performs similarly or sometimes worse than Auto-scheduler. This is because a larger search space requires more time before the cost model can provide an accurate prediction. Deploying an inaccurate cost model on Foresee tuning may result in spending time budget on non-improvement code transformations.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Auto-Scheduler generates a large enough search space, so searching the space with efficiency is important. With FamilySeer, Users can search for the same optimal code under less time budget. We hope that our search method can be an alternative option for those who expect to obtain better optimal code under a limited time budget.

# Prior art
[prior-art]: #prior-art

Please refer to [this paper](https://arxiv.org/abs/2201.00194).

# Unresolved questions
[unresolved-questions]: #unresolved-questions

Our search method is up for [discussion](https://discuss.tvm.apache.org/t/rfc-familyseer-a-new-search-method-for-auto-scheduler/11877).

# Future possibilities
[future-possibilities]: #future-possibilities

1. Dynamic subgraph family anlaysis

FamilySeer currently relies on static analysis to identify subgraphs, which might result in misjudgments on some of the subgraphs. We are looking for an alternative method to identify subgraphs dynamically while maintaining the same time budget.

2. Advanced Foresee tuning

Auto-tuning is the procedure of looking for the only best optimal code for deep learning network. Many less performed codes have to be evaluated to build an accurate cost model. By accurately analyzing the subgraph similarity, we can draw a relationship map between each subgraph and focus on building highly accurate cost model for the most related subgraphs. Once an accurate cost model has been built, We can predict optimal code for other subgraphs instead of searching iteratively.