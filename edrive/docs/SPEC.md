# Interface contract (v0)


- Units: SI everywhere.
- Timebase: global `Ts` in config.
- Signs: power into a block is positive (`P = v*i`).


### Base interfaces
- `PlantBase.step(x, u, dt) -> StepResult(x_next, y, ports)`
- `ControllerBase.update(ref, meas, t) -> u`
- `ObserverBase.update(y, u, t) -> xhat`


### Ports
- `PowerPort`: `{ v: float, i: float }` representing one electrical port.