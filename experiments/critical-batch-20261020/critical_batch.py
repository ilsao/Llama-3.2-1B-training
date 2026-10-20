"""Pure token-window planning and conservative gradient-noise batch control."""
import math
from dataclasses import dataclass, asdict

Q = 131072  # 8 ranks x microbatch2 x8192 nonpadding inputs


def noise_moments(weighted_local_norm2, mean_norm2, targets, world):
    """Unequal-size local means: w_r=n_r/N; E[L-M]=(K-1) trS/N.

    Units are valid prediction tokens. Correlation within documents means the
    result is a noise proxy, not a claim that tokens are independent examples.
    Keep signed signal estimates; smooth moments BEFORE taking their ratio.
    """
    if targets <= 0 or world < 2:
        raise ValueError('Need positive targets and at least two independent ranks')
    noise = targets * (weighted_local_norm2 - mean_norm2) / (world - 1)
    signal = mean_norm2 - noise / targets
    return signal, noise


@dataclass
class Controller:
    mode: str = 'adaptive'
    fraction: float = 0.5
    beta: float = 0.9
    minimum_observations: int = 8
    hysteresis: float = 0.2
    max_units: int = 8
    units: int = 1
    observations: int = 0
    signal_ema: float = 0.0
    noise_ema: float = 0.0
    targets_per_sequence_ema: float = 0.0

    def __post_init__(self):
        if self.mode not in ('adaptive', 'shadow') or not 0 < self.beta < 1:
            raise ValueError('Unsupported mode or EMA beta')
        if self.fraction <= 0 or self.max_units not in (1, 2, 4, 8) or self.units not in (1, 2, 4, 8):
            raise ValueError('Invalid batch control bounds')

    def observe(self, signal, noise, targets_per_sequence, *, allow_change=True):
        old = self.units
        finite = all(math.isfinite(x) for x in (signal, noise, targets_per_sequence))
        if not finite or noise < 0 or targets_per_sequence <= 0:
            return dict(valid=False, reason='invalid_moments', units=old, changed=False)
        self.observations += 1
        for name, value in [('signal_ema', signal), ('noise_ema', noise),
                            ('targets_per_sequence_ema', targets_per_sequence)]:
            setattr(self, name, self.beta * getattr(self, name) + (1-self.beta)*value)
        ready = self.observations >= self.minimum_observations and self.signal_ema > 0
        scale_tokens = self.noise_ema / self.signal_ema if self.signal_ema > 0 else None
        correction = 1 - self.beta ** self.observations
        scale_sequences = (scale_tokens / (self.targets_per_sequence_ema/correction)
                           if scale_tokens is not None else None)
        target = self.fraction * scale_sequences if scale_sequences is not None else None
        if self.mode == 'adaptive' and allow_change and ready:
            choices = [n for n in (1, 2, 4, 8) if n <= self.max_units]
            proposed = min(choices, key=lambda n: abs(math.log(16*n/ max(target, 1e-30))))
            # Require crossing a guarded geometric midpoint; at most2x per probe.
            if proposed > old and target > 16*old*math.sqrt(2)*(1+self.hysteresis):
                self.units = min(old*2, self.max_units)
            elif proposed < old and target < 16*old/math.sqrt(2)/(1+self.hysteresis):
                self.units = max(old//2, 1)
        return dict(valid=True, ready=ready, signal_ema=self.signal_ema/correction,
                    noise_ema=self.noise_ema/correction, noise_scale_target_tokens=scale_tokens,
                    noise_scale_sequence_equivalent=scale_sequences, target_sequences=target,
                    units=self.units, changed=self.units != old)

    def state_dict(self):
        return asdict(self)


def window(seen, step, desired_units, events, target):
    if seen % Q or target % Q or desired_units not in (1, 2, 4, 8):
        raise ValueError('Windows must align to full8rank microbatch2 quanta')
    boundary = min([target] + [x for x in events if x > seen])
    units = min(desired_units, (boundary-seen)//Q)
    if units < 1:
        raise ValueError('No complete training window remains')
    return dict(update=step+1, start_tokens=seen, end_tokens=seen+units*Q,
                width=16*units, accumulation=units)
