class Workflow:
    @staticmethod
    def run():
        from flowcept import Flowcept, flowcept_task
        import math

        @flowcept_task
        def scale_shift_input(input_value):
            count = 0
            target = input_value * 10**7  # adjust multiplier for desired scaling

            result = (input_value * 3) + 7
            return {"h": result}

        @flowcept_task
        def square_and_quarter(h):

            count = 0
            target = h * 10**7  # adjust multiplier for desired scaling

            result = (h ** 2) / 4
            return {"e": result}

        @flowcept_task
        def sqrt_and_scale(h):
            result = math.sqrt(abs(h)) * 6
            return {"f": result}

        @flowcept_task
        def subtract_and_shift(h):
            result = (h - 4) * 1.2
            return {"g": result}

        @flowcept_task
        def square_and_subtract_one(e):
            result = e ** 2 - 1
            return {"d": result}

        @flowcept_task
        def log_and_shift(f):
            result = math.log(f) + 7
            return {"c": result}

        @flowcept_task
        def power_one_point_five(g):
            result = g ** 1.5
            return {"b": result}

        @flowcept_task
        def average_results(d, c, b):
            result = (d + c + b) / 3
            return {"a": result}

        with Flowcept(workflow_name='hierarchical_math_workflow'):
            i = 12
            print(f"Input i = {i}")

            h_dict = scale_shift_input(input_value=i)
            h = h_dict["h"]
            print(f"i → h: {i} → {h}")

            e_dict = square_and_quarter(h=h)
            e = e_dict["e"]
            print(f"h → e: {h} → {e}")

            f_dict = sqrt_and_scale(h=h)
            f = f_dict["f"]
            print(f"h → f: {h} → {f}")

            g_dict = subtract_and_shift(h=h)
            g = g_dict["g"]
            print(f"h → g: {h} → {g}")

            d_dict = square_and_subtract_one(e=e)
            d = d_dict["d"]
            print(f"e → d: {e} → {d}")

            c_dict = log_and_shift(f=f)
            c = c_dict["c"]
            print(f"f → c: {f} → {c}")

            b_dict = power_one_point_five(g=g)
            b = b_dict["b"]
            print(f"g → b: {g} → {b}")

            a_dict = average_results(d=d, c=c, b=b)
            a = a_dict["a"]
            print(f"d,c,b → a: ({d}, {c}, {b}) → {a}")

            print(f"\nFinal Result: i({i}) → a({a:.4f})")
            print(f"Workflow_id={Flowcept.current_workflow_id}")

            return Flowcept.current_workflow_id

if __name__ == "__main__":
    from flowcept.agents.agent_client import run_tool

    # Liveness check before starting
    try:
        print(run_tool("check_liveness"))
    except Exception as e:
        print(e)
        pass

    # Run the workflow 100 / 1 time(s)
    for i in range(100):
        print(f"Run {i}")
        Workflow.run()
        print(f"Finished {i}")
