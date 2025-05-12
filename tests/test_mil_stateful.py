import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb, types

@mb.program(input_specs=[mb.TensorSpec((1,), dtype=types.fp16), 
                         mb.StateTensorSpec((1,), dtype=types.fp16),],)
def prog(x, accumulator_state):
    # Read state... below copy .... "Read a state, copy its content into a new variable"
    accumulator_value = mb.read_state(input=accumulator_state)
    # Update value, also copy .... "Copy the content of a variable into a state and return the copy of the variable"
    y = mb.add(x=x, y=accumulator_value, name="y")
    # Write state
    mb.coreml_update_state(state=accumulator_state, value=y)

    return y

mlmodel = ct.convert(prog,minimum_deployment_target=ct.target.iOS18)