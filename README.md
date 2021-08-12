# IMM-PDA
In this algorithm the Interacting Multiple Models algorithm (IMM) with the Extended Kalman filter (EKF) is combined with the Probabilistic data association filter (PDA) to create an IMM-PDA filter. IMM combines several modes, in our case a Constant Velocity process model (CV-model) and a Constant Turn-rate process model (CT-model). The PDA uses the IMM filter together with information such as the clutter intensity, detection probability and gate size. The final IMM-PDA filter is tuned to accurately fuse position sensor data with the process models for datasets such as the "Joyride" dataset.

## Running
The algorithm can be ran on the command line using
```
python3 run_imm_pda.py
```
## Report
Check out report.pdf for more details
