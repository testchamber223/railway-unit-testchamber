"""Coach thermal model: envelope (k-value), solar, passenger and equipment loads, and the cabin air/thermal mass balance.

No physics implemented yet. Every constant must be obtained via
``evidence.registry.get("E-xxx")`` -- never hard-coded.
"""
