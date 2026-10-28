"""Control logic: setpoint curves, thermostat/PID behaviour and operating-mode sequencing of the HVAC unit.

No physics implemented yet. Every constant must be obtained via
``evidence.registry.get("E-xxx")`` -- never hard-coded.
"""
