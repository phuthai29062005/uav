from pymoo.util.ref_dirs import get_reference_directions

def reference(n_obj, n_partitions):
    # FIX (BUG-1): get_reference_directions co chu ky (name, *args, **kwargs)
    # -- truyen n_partitions theo VI TRI khong bind vao tham so n_partitions
    # ben trong Das-Dennis generator, crash ngay voi
    # "Exception: Either provide number of partitions or number of points."
    # Phai truyen bang KEYWORD.
    return get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
