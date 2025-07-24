from flowcept.agents.agent_client import run_tool


class Workflow:
    @staticmethod
    def run():
        from flowcept import Flowcept, flowcept_task
        import math
        @flowcept_task
        def scale_shift_input(input_value):
            return (input_value * 3) + 7
        @flowcept_task
        def square_and_quarter(h_value):
            return (h_value ** 2) / 4
        @flowcept_task
        def sqrt_and_scale(h_value):
            return math.sqrt(abs(h_value)) * 6
        @flowcept_task
        def subtract_and_shift(h_value):
            return (h_value - 4) * 1.2
        @flowcept_task
        def square_and_subtract_one(e_value):
            return e_value ** 2 - 1
        @flowcept_task
        def log_and_shift(f_value):
            return math.log(f_value) + 7
        @flowcept_task
        def power_one_point_five(g_value):
            return g_value ** 1.5
        @flowcept_task
        def average_results(d_value, c_value, b_value):
            return (d_value + c_value + b_value) / 3

        with Flowcept(workflow_name='hierarchical_math_workflow', start_persistence=False, save_workflow=False):
            I = 12; print(f"Input I = {I}")
            H = scale_shift_input(input_value=I); print(f"I → H: {I} → {H}")
            E, F, G = square_and_quarter(h_value=H), sqrt_and_scale(h_value=H), subtract_and_shift(h_value=H);
            print(f"H → E: {H} → {E}"); print(f"H → F: {H} → {F}"); print(f"H → G: {H} → {G}")
            D, C, B = square_and_subtract_one(e_value=E), log_and_shift(f_value=F), power_one_point_five(g_value=G);
            print(f"E → D: {E} → {D}"); print(f"F → C: {F} → {C}"); print(f"G → B: {G} → {B}")
            A = average_results(d_value=D, c_value=C, b_value=B);
            print(f"D,C,B → A: ({D}, {C}, {B}) → {A}")
            print(f"\nFinal Result: I({I}) → A({A:.4f})")
            print(f"Workflow_id={Flowcept.current_workflow_id}")
            return Flowcept.current_workflow_id

try:
    print(run_tool("check_liveness"))
except Exception as e:
    print(e)
    pass

try:
    print(run_tool("prompt_handler", kwargs={"message": "reset context"}))
except Exception as e:
    print(e)
    pass

for i in range(90):
    print(i)
    Workflow.run()
    print(f"Finished {i}")

try:
    print(run_tool("prompt_handler", kwargs={"message": "save current df"}))
except Exception as e:
    print(e)
    pass
