import warp as wp


@wp.struct
class Dd_dx:
    dd_dx0: wp.vec3
    dd_dx1: wp.vec3
    dd_dx2: wp.vec3
    dd_dx3: wp.vec3
    

@wp.func
def example_func(x0: wp.vec3, x1: wp.vec3, x2: wp.vec3, x3: wp.vec3) -> Dd_dx:
    dd_dx = Dd_dx()
    dd_dx.dd_dx0 = x0 - x1
    dd_dx.dd_dx1 = x0 - x2
    dd_dx.dd_dx2 = x0 - x3
    dd_dx.dd_dx3 = x0 - x0
    return dd_dx


@wp.kernel
def example_kernel(
    structs: wp.array(dtype=Dd_dx),
):
    tid = wp.tid()
    
    x0 = wp.vec3(0.0, 0.0, 0.0)
    x1 = wp.vec3(1.0, 1.0, 1.0)
    x2 = wp.vec3(2.0, 2.0, 2.0)
    x3 = wp.vec3(3.0, 3.0, 3.0)
    
    structs[tid] = example_func(x0, x1, x2, x3)
    

wp.init()
device = wp.get_preferred_device()
structs = wp.zeros(4, dtype=Dd_dx)
wp.launch(
    kernel=example_kernel, 
    dim=1, 
    inputs=[structs],
    device=device, 
)
print(structs.numpy())
