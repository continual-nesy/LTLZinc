import minizinc
import re

class StateSampler:
    """
    Solve the associated MiniZinc problem for each state, then sample random solutions.
    """
    def __init__(self, domains, types, mapping, rng, prefix="", solver="gecode"):
        self.solver = minizinc.Solver.lookup(solver)
        self.rng = rng

        self.mapping = mapping
        self.types = {k: ({kk for kk in v.keys()} if isinstance(v, dict) else set(v)) for k, v in types.items()} # No need for path information.
        self.domains = domains

        self.props = sorted(mapping.keys())
        self.vars = sorted(self.domains.keys())

        self.prefix = prefix
        self.solution_cache = {}

    def _substitute(self, constraint, substitutions):
        """
        Replace place-holders with variables in the constraint definition.
        :param constraint: A string containing place-holders.
        :param substitutions: A dictionary of substitutions.
        :return: A string containing substituted variables.
        """
        out = constraint
        for k, v in substitutions.items():
            out = out.replace(k, v)
        return out

    def _sympy_to_minizinc(self, formula):
        """
        Convert SymPy boolean formulas to MiniZinc constraints.
        :param formula: A string in SymPy format.
        :return: A string in MiniZinc format.
        """
        return formula.replace("~", "not ").replace("|", "\/").replace("&", "/\\")

    def solve(self, formula, solutions=1, split="train"):
        """
        Solve the given transition formula (set of constraints) and sample the desired number of solutions.
        :param formula: A string describing a transition formula.
        :param solutions: Number of solutions to output.
        :param split: The dataset split, to select the correct domain for each variable.
        :return: Tuple (list of variable assignments for each solution, list of constraint labels for each solution).
        """
        atoms = re.findall(r'p_[0-9]+', str(formula))
        cache_key = str(formula) # Note: correct, but inefficient in case of equivalent formulas/permutations (e.g. "p_0 & !p_1" vs "!p_1 & p_0").
        # Storing only the (sorted) atoms or literals is wrong
        # (i.e. "p_0" would collide with "!p_0" in the first case, and "p_0 & !p_1" would collide with "p_0 | !p_1" in the second).


        if cache_key not in self.solution_cache:

            problem = self.prefix
            problem += "\n"
            for p in self.props:
                problem += "var bool: {};\n".format(p)
            problem += "\n"

            numeric_types = set()

            # Collect all the non-numeric values.
            global_enum = {kk for v in self.types.values() for kk in v if not isinstance(kk, int)}
            problem += "enum global_enum_t = {{{}}};\n\n".format(", ".join(sorted(global_enum)))

            # Associate the correct domain, with respect to the current dataset split.
            for k, v in self.types.items():
                if isinstance(list(v)[0], int):
                    numeric_types.add(k)
                else:
                    problem += "set of global_enum_t: {} = {{{}}};\n".format(k, ", ".join(sorted(v)))

            problem += "\n"

            # Define each variable.
            for k, v in self.domains.items():
                if v[split] in numeric_types:
                    problem += "var {{{}}}: {};\n".format(", ".join([str(vv) for vv in sorted(self.types[v[split]])]), k)
                else:
                    problem += "var {}: {};\n".format(v[split], k)

            problem += "\n"

            # Define reified constraints.
            for k, v in self.mapping.items():
                problem += "constraint {} <-> ({});\n".format(k, self._substitute(v["constraint"], v["substitutions"]))

            # Define current transition constraint.
            if str(formula) != "True":
                problem += "\nconstraint {};\n".format(self._sympy_to_minizinc(str(formula)))

            problem += "solve satisfy;"

            model = minizinc.Model()
            model.add_string(problem)

            instance = minizinc.Instance(self.solver, model)

            # Solve the problem and cache all the solutions, in case the same transition is encountered again in the future.
            result = instance.solve(all_solutions=True)
            if not result.status.has_solution():
                raise RuntimeError("The following problem has no solution:\n{}".format(model._code_fragments))

            self.solution_cache[cache_key] = result.solution # Memoization of solutions, to avoid recomputing them.

        rnd_sol = self.rng.choice(self.solution_cache[cache_key], size=solutions)
        labels = []
        assignments = []
        for i in range(solutions):
            labels.append({})
            assignments.append({})

            # Annotate the chosen random solution.
            # Convention: 0/1 -> constraint is known to be False/True.
            # (0)/(1) -> constraint is irrelevant for the current transition, but, given the chosen variable assignments, it was False/True.
            for k, v in rnd_sol[i].__dict__.items():
                if not k.startswith("_"):
                    if k.startswith("p_"):
                        if k in atoms and cache_key != "True":
                            labels[-1][k] = "1" if v else "0"
                        else:
                            # Don't care since this literal does not appear in the formula, but still label it.
                            labels[-1][k] = "(1)" if v else "(0)"
                    else:
                        # Variables are assigned to their corresponding solution, whether they were relevant or not for the current transition.
                        assignments[-1][k] = v

        return assignments, labels