# Risk Quant Calculator

A risk calculation tool for approaching cybersecurity risk using statistical methods.

## Overview

This tool is designed for performing quantitative risk assessments and analyzing the monetary impact of cybersecurity risks, optionally incorporating the effectiveness of risk controls. This tool is designed to be used on the command line but its output can be easily integrated into other applications through the use of JSON.

`risk_simulator.py` - Uses Monte Carlo simulation to:
- Model potential losses based on asset value, exposure factor, and rate of occurrence
- Analyze effectiveness of risk controls through before/after comparisons
- Generate risk distributions and loss exceedance curves
- Calculate key risk statistics and percentiles

## Installation

Requirements:
```bash
pip install -r requirements.txt
```

## Usage

```bash
python risk_simulator.py
```

## To Do

- [X] Add Monte Carlo simulation
- [X] Add mean, median, mode, standard deviation, and percentiles
- [X] Add loss exceedance curve
- [X] Add before/after risk control analysis
- [ ] Better way to input data
- [ ] Add more risk control analysis options (e.g. bayesian analysis)
- [ ] Add more risk statistics (e.g. Value at Risk)
