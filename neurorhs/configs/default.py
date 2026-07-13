from neurorhs.neurosci import *
import diffrax


class FooConfig:
    def __init__(self, context, default_arguments, is_dynamic):
        self.ctx = context
        self.default_arguments = default_arguments
        self.is_dynamic = is_dynamic
        self.dynamic_part, self.static_part = get_dynamic_static_parts(default_arguments, is_dynamic)

        def setup(y):
            s = combine_parts(y, self.static_part, self.is_dynamic)
            ds_dt = jax.tree_util.tree_map(jnp.zeros_like, y)
            return s, ds_dt
    
        self.setup = setup

    def get_f_explicit(self):
        raise NotImplementedError

    def get_f_implicit(self):
        raise NotImplementedError

    def get_combined(self):
        def foo(t, y, args):
            return self.get_f_explicit(t, y, args) + self.get_f_implicit(t, y, args)
        return foo


class SimulationConfig:
    def __init__(self, foo_config:FooConfig):
        self.foo_config:FooConfig = foo_config
        self.f_explicit = foo_config.get_f_explicit()
        self.f_implicit = foo_config.get_f_implicit()

        self.term_explicit = diffrax.ODETerm(self.f_explicit)
        self.term_implicit = diffrax.ODETerm(self.f_implicit)
        self.terms = diffrax.MultiTerm(self.term_explicit, self.term_implicit)

    def get_solver(self):
        raise NotImplementedError

    def get_stepsize_controller(self):
        raise NotImplementedError

    def solve(self, t0, t1, dt0 = 0.01, y0 = None, num = 100, max_steps = 100_000):
        if y0 is None:
            y0 = self.foo_config.dynamic_part

        sol = diffrax.diffeqsolve(
            self.terms,
            self.get_solver(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            stepsize_controller=self.get_stepsize_controller(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, num)),
            max_steps = max_steps
        )
        return sol