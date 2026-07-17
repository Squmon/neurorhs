import neurorhs.io as io
from neurorhs.neurosci import *
import diffrax
import lineax as lx
import optimistix as optx


class FooConfig:
    def __init__(self, context, default_arguments, is_dynamic, groups):
        self.ctx = context
        self.default_arguments = default_arguments
        self.is_dynamic = is_dynamic
        self.groups = groups

    def get_dynamic_static_parts(self, state = None):
        if state is None:
            state = self.default_arguments
        return get_dynamic_static_parts(state, self.is_dynamic)

    def construct_f_explicit(self):
        return lambda s, ds_dt, t: ds_dt

    def construct_f_implicit(self):
        return lambda s, ds_dt, t: ds_dt

    def get_diffrax_explicit_part(self):
        fe = self.construct_f_explicit()
        _, s_part = self.get_dynamic_static_parts()

        def f_exp(t, y, args):
            s = combine_parts(y, s_part, self.is_dynamic)
            ds_dt = jax.tree_util.tree_map(jnp.zeros_like, y)
            ds_dt = fe(s, ds_dt, t)
            return ds_dt

        return f_exp

    def get_diffrax_implicit_part(self):
        fe = self.construct_f_implicit()
        _, s_part = self.get_dynamic_static_parts()

        def f_imp(t, y, args):
            s = combine_parts(y, s_part, self.is_dynamic)
            ds_dt = jax.tree_util.tree_map(jnp.zeros_like, y)
            ds_dt = fe(s, ds_dt, t)
            return ds_dt

        return f_imp

    def get_combined(self):
        def foo(t, y, args):
            fe, fi = self.get_f_explicit(), self.get_f_implicit()
            return jax.tree_util.tree_map(lambda x, y: x+y, fe(t, y, args), fi(t, y, args))
        return foo


class SimulationConfig:
    def __init__(self, foo_config: FooConfig):
        self.foo_config: FooConfig = foo_config

        self.f_explicit = foo_config.get_diffrax_explicit_part()
        self.f_implicit = foo_config.get_diffrax_implicit_part()

        self.term_explicit = diffrax.ODETerm(self.f_explicit)
        self.term_implicit = diffrax.ODETerm(self.f_implicit)
        self.terms = diffrax.MultiTerm(self.term_explicit, self.term_implicit)

    def get_solver(self):
        raise NotImplementedError

    def get_stepsize_controller(self):
        raise NotImplementedError

    def solve(self, t0, t1, dt0=0.01, y0=None, num=100, max_steps=100_000, save_at=None):
        if save_at is None:
            save_at = diffrax.SaveAt(ts=jnp.linspace(t0, t1, num))
        if y0 is None:
            y0, _ = self.foo_config.get_dynamic_static_parts()

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



    def save_results(self, sol, path, **kwargs):
        _, static_part = self.foo_config.get_dynamic_static_parts()
        io.save_pytree({
            "ys":sol.ys,
            "ts":sol.ts,
            'static_part':static_part,
            "is_dynamic":self.foo_config.is_dynamic,
            "mapping":self.foo_config.ctx['mapping']
        }, path, **kwargs)

    def load_results(self, path, to_jax = False):
        result = io.load_pytree(path, to_jax)
        return result

    def load_results_with_mapping(path, to_jax = False):
        raise NotImplementedError

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
