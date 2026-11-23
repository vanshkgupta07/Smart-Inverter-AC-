"""
Flask Web Application for Soft Computing Fuzzy Logic Project
Bidirectional Fuzzy Inference: Fuzzification <---> Defuzzification
"""

import os
from flask import Flask, render_template, request, jsonify
from fuzzy_engine import SmartACFuzzySystem

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
app = Flask(__name__, template_folder=template_dir)
system = SmartACFuzzySystem()

@app.route("/")
@app.route("/api/index")
@app.route("/api/index.py")
def index():
    return render_template("index.html")

@app.route("/api/compute", methods=["POST", "GET"])
@app.route("/compute", methods=["POST", "GET"])
def compute():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        temp = float(data.get("temp", 32.0))
    else:
        temp = float(request.args.get("temp", 32.0))

    # 1. Forward Pipeline (Crisp -> Fuzzify -> Defuzzify)
    forward_res = system.forward_pipeline(temp)
    crisp_power = forward_res["defuzzification"]["centroid"]

    # 2. Defuzzification -> Fuzzification (Output Re-Fuzzification)
    reverse_res = system.reverse_refuzzify(crisp_power)

    # 3. Dynamic 1-Step Feedback (Plant dynamics -> New State -> Fuzzify)
    feedback_res = system.closed_loop_step(temp, crisp_power)

    return jsonify({
        "status": "success",
        "forward": forward_res,
        "reverse": reverse_res,
        "feedback": feedback_res
    })

@app.route("/api/simulate", methods=["POST", "GET"])
@app.route("/simulate", methods=["POST", "GET"])
def simulate():
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form
        init_temp = float(data.get("temp", 38.0))
        steps = int(data.get("steps", 12))
    else:
        init_temp = float(request.args.get("temp", 38.0))
        steps = int(request.args.get("steps", 12))

    history = system.simulate_closed_loop(initial_temp=init_temp, steps=steps)
    return jsonify({
        "status": "success",
        "steps": history
    })

@app.route("/api/curves", methods=["GET"])
@app.route("/curves", methods=["GET"])
def get_curves():
    # Generate points for input temperature MF curves
    temp_points = [x * 0.5 for x in range(0, 101)] # 0 to 50 C
    temp_curves = {}
    for term_name, mf in system.temp_var.terms.items():
        temp_curves[term_name] = [round(mf.evaluate(x), 3) for x in temp_points]

    # Generate points for output power MF curves
    power_points = [y for y in range(0, 101)] # 0 to 100%
    power_curves = {}
    for term_name, mf in system.power_var.terms.items():
        power_curves[term_name] = [round(mf.evaluate(y), 3) for y in power_points]

    return jsonify({
        "temp_x": temp_points,
        "temp_curves": temp_curves,
        "power_x": power_points,
        "power_curves": power_curves
    })

if __name__ == "__main__":
    print("=" * 60)
    print(" Starting Fuzzy Logic Controller Web App on http://127.0.0.1:5000")
    print(" Open your browser and navigate to http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
