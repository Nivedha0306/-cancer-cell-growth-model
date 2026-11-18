
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.set_page_config(page_title="Cancer Cell Modelling", layout="wide")

st.title("Cancer Cell Growth and Drug Treatment Response")
st.write("Systems Biology-Based Modelling")

st.sidebar.header("Simulation Parameters")

N0 = st.sidebar.number_input("Initial Cancer Cells", 10, 10000, 100)
r = st.sidebar.slider("Growth Rate", 0.01, 0.5, 0.2)
K = st.sidebar.number_input("Carrying Capacity", 1000, 100000, 10000)
alpha = st.sidebar.slider("Drug Effectiveness", 0.001, 0.1, 0.05)
D = st.sidebar.slider("Drug Concentration", 0.0, 10.0, 5.0)
days = st.sidebar.slider("Simulation Days", 5, 100, 30)

def model(t, y, drug):
    N = max(y[0], 0)
    return [r*N*(1-N/K) - alpha*drug*N]

t = np.linspace(0, days, 200)

def simulate(drug):
    result = solve_ivp(model, [0, days], [N0],
                       args=(drug,), t_eval=t)
    return result.y[0]

untreated = simulate(0)
treated = simulate(D)

st.subheader("Cancer Cell Growth Comparison")

fig, ax = plt.subplots()
ax.plot(t, untreated, label="Without Drug")
ax.plot(t, treated, label="With Drug")
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Cancer Cell Population")
ax.legend()
ax.grid()
st.pyplot(fig)
plt.close(fig)

st.subheader("Drug Concentration Comparison")

fig, ax = plt.subplots()

for dose in [0, 2, 5, 8]:
    ax.plot(t, simulate(dose), label=f"Drug {dose}")

ax.set_xlabel("Time (Days)")
ax.set_ylabel("Cancer Cell Population")
ax.legend()
ax.grid()
st.pyplot(fig)
plt.close(fig)

st.subheader("Simulation Results")

results = pd.DataFrame({
    "Treatment": ["No Drug", "Drug 2", "Drug 5", "Drug 8"],
    "Final Cell Population": [
        simulate(0)[-1], simulate(2)[-1],
        simulate(5)[-1], simulate(8)[-1]
    ]
})

st.dataframe(results)

reduction = (1 - treated[-1]/untreated[-1])*100
st.write(f"Estimated cell population reduction: {reduction:.2f}%")

st.info("Illustrative mathematical simulation, not a clinical prediction.")
