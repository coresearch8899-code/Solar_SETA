# Data dictionary

Both files are CT-corrected (grid import and export divided by 1.20), sorted by time (and zone) on a complete 30-min grid
from 2023-03-03 00:00 to 2026-09-07 23:30 local time (UTC+7). Missing values are empty. Timestamps mark the end of the interval.

## data/zone_30min_series.csv (185,040 rows = 61,680 intervals x 3 zones)
| Column | Description | Unit | Notes |
|---|---|---|---|
| interval_end_local_time | End of the 30-min interval | YYYY-MM-DD hh:mm | |
| zone | Zone number | - | 1, 2, 3 |
| zone_kWp | Installed PV capacity of the zone | kWp | 2256.24, 207.06, 49.90 |
| pv_kWh | PV energy, E | kWh | Difference of lifetime counters; empty if rejected by QC |
| gross_load_kWh | Gross load, L = E + B - X | kWh | From corrected B and X |
| grid_import_kWh | Grid import, B | kWh | Portal value / 1.20 |
| export_kWh | Export, X | kWh | Portal value / 1.20 |
| ghi_W_m2 | Global horizontal irradiance of the zone sensor | W/m2 | Night-time gaps set to 0 |
| ambient_temp_C | Ambient temperature of the zone sensor | degC | |
| pv_power_kW | Instantaneous PV power reported by the portal | kW | |
| plant_status | Portal status | - | normal, alarm, fault |

## data/hospital_30min_series.csv (61,680 rows)
| Column | Description | Unit | Notes |
|---|---|---|---|
| interval_end_local_time | End of the 30-min interval | YYYY-MM-DD hh:mm | |
| pv_kWh, gross_load_kWh, grid_import_kWh, export_kWh | Hospital totals of the zone values | kWh | Only when all three zones passed QC; export = 0.10% of PV |
| grid_import_portal_raw_kWh | Grid import as recorded by the portal | kWh | Uncorrected (sensitivity case, invoice check) |
| ambient_temp_C, ghi_W_m2 | Median of the three zone sensors | degC, W/m2 | |
| non_working_day | Weekend or Thai public holiday | 0/1 | holidays library |

The 60 intervals missing in all three zones (2024-05-31 and 2024-08-07) are empty rows. The 20 hospital-level
intervals that form counter-timing pairs (energy logged one interval late; Section S1, rule v) are kept in both files
and are rejected in step 01 of the code (`code/qc_timing.py`, listed in `output/tab_qc_timing.csv`).

## data/monthly_invoices_main_account.csv (45 rows, January 2023 to September 2026)
| Column | Description | Unit |
|---|---|---|
| billing_month | Billing month (meter read at month end) | YYYY-MM |
| invoiced_energy_kWh | Billed energy of the main account | kWh |
| billed_demand_kW | Billed 15-min on-peak demand | kW |
| invoice_total_THB | Invoice total incl. Ft and 7% VAT | THB |

Account identifiers are omitted.
