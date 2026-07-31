import sys

class MasterExecutionScript:
    def __init__(self):
        self.registry = {}
        self.execution_order = []

    def register_task(self, name, task_callable):
        if not callable(task_callable):
            raise TypeError()
        self.registry[name] = task_callable
        self.execution_order.append(name)

    def run_all(self, context=None):
        results = {}
        for task_name in self.execution_order:
            try:
                task_func = self.registry[task_name]
                results[task_name] = task_func(context) if context else task_func()
            except Exception as e:
                print(f"Task {task_name} failed", file=sys.stderr)
                results[task_name] = None
        return results