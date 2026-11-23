"""
Fuzzy Engine - Core Soft Computing Implementation
Implements:
1. Crisp -> Fuzzification (Triangular & Trapezoidal Membership Functions)
2. Mamdani Fuzzy Inference Engine (Min-Max Composition)
3. Defuzzification (Centroid / Center of Gravity & Mean of Maxima)
4. Defuzzification -> Fuzzification:
   - Direct Re-Fuzzification of Actuator Output
   - Closed-Loop Feedback Control (Defuzzified Output alters Plant state -> Re-Fuzzified)
   - Inverse Fuzzy Inference (Given target output -> deduce input)
"""

from typing import Dict, List, Tuple
import math

class MembershipFunction:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, x: float) -> float:
        raise NotImplementedError


class TriangularMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        elif self.b < x < self.c:
            if self.c == self.b:
                return 1.0
            return (self.c - x) / (self.c - self.b)
        return 0.0


class TrapezoidalMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def evaluate(self, x: float) -> float:
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x <= self.b:
            if self.b == self.a:
                return 1.0
            return (x - self.a) / (self.b - self.a)
        elif self.b < x <= self.c:
            return 1.0
        elif self.c < x < self.d:
            if self.d == self.c:
                return 1.0
            return (self.d - x) / (self.d - self.c)
        return 0.0


class FuzzyVariable:
    def __init__(self, name: str, unit: str, min_val: float, max_val: float, step: float = 0.5):
        self.name = name
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.terms: Dict[str, MembershipFunction] = {}

    def add_term(self, term: MembershipFunction):
        self.terms[term.name] = term

    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        """Convert crisp number to membership values mu_A(x) in [0, 1]"""
        # Clamp to variable range
        clamped = max(self.min_val, min(self.max_val, crisp_value))
        memberships = {}
        for term_name, mf in self.terms.items():
            memberships[term_name] = round(mf.evaluate(clamped), 4)
        return memberships

    def get_universe_points(self) -> List[float]:
        points = []
        curr = self.min_val
        while curr <= self.max_val + 1e-9:
            points.append(round(curr, 2))
            curr += self.step
        return points


class FuzzyRule:
    def __init__(self, rule_id: int, antecedent_term: str, consequent_term: str, description: str = ""):
        self.rule_id = rule_id
        self.antecedent_term = antecedent_term  # e.g. "Hot"
        self.consequent_term = consequent_term  # e.g. "High"
        self.description = description or f"IF Temperature is {antecedent_term} THEN Cooling Power is {consequent_term}"


class SmartACFuzzySystem:
    def __init__(self):
        # 1. Input Variable: Temperature (0 C to 50 C)
        self.temp_var = FuzzyVariable("Temperature", "°C", 0.0, 50.0, step=0.5)
        # Trapezoidal for extreme cold (<=15)
        self.temp_var.add_term(TrapezoidalMF("Freezing", -5, 0, 12, 18))
        self.temp_var.add_term(TriangularMF("Cold", 14, 20, 24))
        self.temp_var.add_term(TriangularMF("Comfortable", 21, 25, 29))
        self.temp_var.add_term(TriangularMF("Warm", 26, 31, 36))
        # Trapezoidal for extreme hot (>=38)
        self.temp_var.add_term(TrapezoidalMF("Hot", 33, 40, 50, 55))

        # 2. Output Variable: AC Cooling Power (0% to 100%)
        self.power_var = FuzzyVariable("CoolingPower", "%", 0.0, 100.0, step=1.0)
        self.power_var.add_term(TrapezoidalMF("Off/Min", -10, 0, 10, 25))
        self.power_var.add_term(TriangularMF("Low", 15, 30, 45))
        self.power_var.add_term(TriangularMF("Medium", 35, 55, 75))
        self.power_var.add_term(TriangularMF("High", 65, 80, 95))
        self.power_var.add_term(TrapezoidalMF("Turbo", 85, 95, 100, 110))

        # 3. Rule Base (Knowledge Base)
        self.rules = [
            FuzzyRule(1, "Freezing", "Off/Min", "IF Temp is Freezing -> Cooling Power is Off/Min (Save Energy)"),
            FuzzyRule(2, "Cold", "Low", "IF Temp is Cold -> Cooling Power is Low"),
            FuzzyRule(3, "Comfortable", "Medium", "IF Temp is Comfortable -> Cooling Power is Medium (Maintain)"),
            FuzzyRule(4, "Warm", "High", "IF Temp is Warm -> Cooling Power is High"),
            FuzzyRule(5, "Hot", "Turbo", "IF Temp is Hot -> Cooling Power is Turbo (Rapid Cooling)")
        ]

    # --- FORWARD PIPELINE: FUZZIFICATION -> DEFUZZIFICATION ---
    def forward_pipeline(self, crisp_temp: float) -> dict:
        """
        Takes crisp temperature, performs:
        1. Fuzzification
        2. Rule Firing & Mamdani Inference (Min-Max)
        3. Defuzzification (Centroid & Mean of Maximum)
        """
        # Step 1: Fuzzification
        fuzzy_inputs = self.temp_var.fuzzify(crisp_temp)

        # Step 2: Rule Evaluation (Mamdani Min operator)
        rule_firings = []
        for rule in self.rules:
            strength = fuzzy_inputs.get(rule.antecedent_term, 0.0)
            rule_firings.append({
                "rule_id": rule.rule_id,
                "antecedent": rule.antecedent_term,
                "consequent": rule.consequent_term,
                "description": rule.description,
                "firing_strength": strength
            })

        # Step 3: Aggregation over the output universe (Mamdani Max operator)
        output_universe = self.power_var.get_universe_points()
        aggregated_membership = []
        rule_clipped_curves = {r.consequent_term: [] for r in self.rules}

        for y in output_universe:
            # calculate clipped membership for each rule: min(firing_strength, MF(y))
            # then take max across all rules for that consequent
            mu_agg = 0.0
            for rf in rule_firings:
                c_term = rf["consequent"]
                alpha = rf["firing_strength"]
                consequent_mf = self.power_var.terms[c_term]
                val = min(alpha, consequent_mf.evaluate(y))
                if val > mu_agg:
                    mu_agg = val
            aggregated_membership.append(round(mu_agg, 4))

        # Step 4: Defuzzification
        # 4a. Centroid Method (Center of Gravity - COG)
        numerator = sum(y * mu for y, mu in zip(output_universe, aggregated_membership))
        denominator = sum(aggregated_membership)
        if denominator > 0:
            centroid_crisp = round(numerator / denominator, 2)
        else:
            centroid_crisp = 0.0

        # 4b. Mean of Maxima (MOM)
        max_mu = max(aggregated_membership) if aggregated_membership else 0.0
        if max_mu > 0:
            max_points = [y for y, mu in zip(output_universe, aggregated_membership) if math.isclose(mu, max_mu, abs_tol=1e-4)]
            mom_crisp = round(sum(max_points) / len(max_points), 2)
        else:
            mom_crisp = 0.0

        return {
            "crisp_input_temp": crisp_temp,
            "fuzzification": fuzzy_inputs,
            "rule_firings": rule_firings,
            "defuzzification": {
                "centroid": centroid_crisp,
                "mom": mom_crisp,
                "numerator": round(numerator, 4),
                "denominator": round(denominator, 4)
            },
            "output_universe": output_universe,
            "aggregated_curve": aggregated_membership
        }

    # --- REVERSE / FEEDBACK PIPELINE: DEFUZZIFICATION -> FUZZIFICATION ---
    def reverse_refuzzify(self, crisp_power: float) -> dict:
        """
        Step 1 of Defuzzification -> Fuzzification:
        Takes a crisp defuzzified output (e.g. 78% Power) and re-evaluates
        it through the output membership functions to classify the actuator
        state into linguistic terms.
        """
        re_fuzzified = self.power_var.fuzzify(crisp_power)
        # Find dominant linguistic state
        dominant_state = max(re_fuzzified.items(), key=lambda item: item[1])
        return {
            "crisp_power": crisp_power,
            "re_fuzzified_memberships": re_fuzzified,
            "dominant_state": dominant_state[0],
            "dominant_degree": dominant_state[1],
            "explanation": f"Crisp output {crisp_power}% maps back to Fuzzy State '{dominant_state[0]}' with {round(dominant_state[1]*100, 1)}% confidence."
        }

    def closed_loop_step(self, current_temp: float, crisp_power: float, ambient_heat: float = 32.0, cooling_rate: float = 0.08) -> dict:
        """
        Step 2 of Defuzzification -> Fuzzification (Real-World Feedback Loop):
        The defuzzified crisp power acts on the physical environment (room).
        Room cools down:
        Delta_T = - cooling_rate * (crisp_power / 100) * 8.0 + 0.02 * (ambient_heat - current_temp)
        New Temperature = Current Temperature + Delta_T
        This new crisp temperature is then IMMEDIATELY FED BACK INTO THE FUZZIFIER!
        """
        cooling_effect = cooling_rate * (crisp_power / 100.0) * 8.0
        ambient_leak = 0.04 * (ambient_heat - current_temp)
        delta_t = -cooling_effect + ambient_leak
        new_temp = round(current_temp + delta_t, 2)
        new_temp = max(5.0, min(48.0, new_temp))

        # Re-fuzzify the newly formed crisp state!
        new_fuzzification = self.temp_var.fuzzify(new_temp)

        return {
            "previous_temp": current_temp,
            "applied_crisp_power": crisp_power,
            "delta_temp": round(delta_t, 3),
            "new_crisp_temp": new_temp,
            "re_fuzzified_new_state": new_fuzzification,
            "dominant_new_state": max(new_fuzzification.items(), key=lambda x: x[1])[0]
        }

    def simulate_closed_loop(self, initial_temp: float, steps: int = 12, ambient_heat: float = 35.0) -> List[dict]:
        """
        Executes multiple closed-loop cycles:
        Temp -> Fuzzify -> Defuzzify -> Power -> New Temp -> Re-Fuzzify -> Repeat!
        """
        history = []
        curr_temp = initial_temp
        for step_idx in range(1, steps + 1):
            # Forward: Fuzzification -> Defuzzification
            fwd = self.forward_pipeline(curr_temp)
            applied_power = fwd["defuzzification"]["centroid"]

            # Output Re-fuzzification
            output_refuzz = self.reverse_refuzzify(applied_power)

            # Environmental Feedback (Defuzz -> New Crisp State -> Fuzzification)
            fb = self.closed_loop_step(curr_temp, applied_power, ambient_heat=ambient_heat)

            history.append({
                "step": step_idx,
                "input_temp": curr_temp,
                "input_fuzzification": fwd["fuzzification"],
                "defuzzified_power": applied_power,
                "output_refuzzification": output_refuzz["re_fuzzified_memberships"],
                "new_temp": fb["new_crisp_temp"],
                "new_temp_fuzzification": fb["re_fuzzified_new_state"]
            })
            curr_temp = fb["new_crisp_temp"]

        return history
