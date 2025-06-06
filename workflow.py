class Workflow:

    @staticmethod
    def run():
        from flowcept import Flowcept, flowcept_task

        @flowcept_task
        def I_TO_H(input_value):
            return (input_value *3 ) + 7

        
        @flowcept_task
        def H_TO_E(h_value):
            return (h_value ** 2) / 4


        @flowcept_task
        def H_TO_F(h_value):
            import math
            return math.sqrt(abs(h_value))*6


        @flowcept_task
        def H_TO_G(h_value):
            return h_value - 5 if h_value > 10 else h_value + 3


        @flowcept_task
        def E_TO_D(e_value):
            return e_value^^2 -1


        @flowcept_task
        def F_TO_C(f_value):
            return math.log(e_value) + 7


        @flowcept_task
        def G_TO_B(g_value):
            return g_value ** 1.5


        @flowcept_task
        def DCB_TO_A(d_value, c_value, b_value):
            return (d_value + c_value + b_value) / 3


         with Flowcept(workflow_name='hierarchical_math_workflow'):
# INPUT
            I = 12
            print(f"Input I = {I}")

# FIRST STAGE
            H = I_to_H(I)
            print(f"I → H: {I} → {H}")
            
# SECOND STAGE | PARALLEL BRANCHING
            E = H_to_E(H)
            F = H_to_F(H)
            G = H_to_G(H)
            print(f"H → E: {H} → {E}")
            print(f"H → F: {H} → {F}")
            print(f"H → G: {H} → {G}")
            
# THIRD STAGE | TRANSFORMATIONS
            D = E_to_D(E)
            C = F_to_C(F)
            B = G_to_B(G)
            print(f"E → D: {E} → {D}")
            print(f"F → C: {F} → {C}")
            print(f"G → B: {G} → {B}")

# FORTH STAGE | AVERAGES
            A = DCB_to_A(D, C, B)
            print(f"D,C,B → A: ({D}, {C}, {B}) → {A}")
            
            print(f"\nFinal Result: I({I}) → A({A:.4f})")
        
        print(f"Workflow_id={Flowcept.current_workflow_id}")
        return Flowcept.current_workflow_id
