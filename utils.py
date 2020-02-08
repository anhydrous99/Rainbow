

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            return func
        return dec(func)
    return decorator


def slice_per(source, step):
    return [source[i::step] for i in range(step)]
