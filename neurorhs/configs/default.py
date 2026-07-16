from neurorhs.neurosci import *
import diffrax
import lineax as lx
import optimistix as optx


class FooConfig:
    def __init__(self, context, default_arguments, is_dynamic):
        self.ctx = context
        self.default_arguments = default_arguments
        self.is_dynamic = is_dynamic
        self.dynamic_part, self.static_part = get_dynamic_static_parts(
            default_arguments, is_dynamic)

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
            fe, fi = self.get_f_explicit(), self.get_f_implicit()
            return jax.tree_util.tree_map(lambda x, y:x+y, fe(t, y, args), fi(t, y, args))
        return foo


class SimulationConfig:
    def __init__(self, foo_config: FooConfig):
        self.foo_config: FooConfig = foo_config
        self.f_explicit = foo_config.get_f_explicit()
        self.f_implicit = foo_config.get_f_implicit()

        self.term_explicit = diffrax.ODETerm(self.f_explicit)
        self.term_implicit = diffrax.ODETerm(self.f_implicit)
        self.terms = diffrax.MultiTerm(self.term_explicit, self.term_implicit)

    def get_solver(self):
        raise NotImplementedError

    def get_stepsize_controller(self):
        raise NotImplementedError

    def solve(self, t0, t1, dt0=0.01, y0=None, num=100, max_steps=100_000, save_at = None):
        if save_at is None:
            save_at = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num))
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
            saveat=save_at,
            max_steps=max_steps
        )
        return sol


class DefaultSim(SimulationConfig):
    def get_stepsize_controller(self):
        return diffrax.PIDController(rtol=1e-5, atol=1e-5)

    def get_solver(self):
        # 1. Линейный решатель остается прежним (GMRES не строит плотную матрицу)
        linear_solver = lx.GMRES(rtol=1e-5, atol=1e-5)

        # 2. Нелинейный решатель теперь создается через optimistix
        root_finder = optx.Newton(
            rtol=1e-5,
            atol=1e-5,
            linear_solver=linear_solver
        )

        solver = diffrax.Sil3(root_finder=root_finder)
        return solver
