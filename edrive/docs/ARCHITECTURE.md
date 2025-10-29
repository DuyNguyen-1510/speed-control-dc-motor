# Architecture (high level)


- **Power chain:** `source → plants/boost → DC bus → actuators/hbridge → plants/motor`.
- **Controllers:** `bus (Vbus)`, `current (Ia)`, `speed (ω)`.
- **Observers:** ESO / SMO for disturbances/torque.
- **Profiles:** composable Voc(t), loads, disturbances.
- **Runtime:** a scheduler advances time; modules exchange **SI units** via typed dataclasses.


All modules implement base interfaces from `edrive.common.base`.