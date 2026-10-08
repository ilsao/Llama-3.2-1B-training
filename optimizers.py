"""Parameter routing for native torch.optim.Muon and torch.optim.AdamW.

Imported only on allocated compute nodes. No optimizer algorithms are defined here.
"""
import torch

def muon_groups(model, cfg):
    """Hidden Linear weights use Muon; embeddings/head/norm/bias use AdamW.

    Module identity exclusions apply to every alias of a tied parameter.
    Unknown non-Linear matrices stay on AdamW with normal decay semantics.
    """
    excluded = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                for p in module.parameters(recurse=False)}
    output = model.get_output_embeddings()
    if output is not None:
        excluded.update(id(p) for p in output.parameters())
    hidden = {id(module.weight) for module in model.modules() if isinstance(module, torch.nn.Linear)
              and id(module.weight) not in excluded}
    parameters = {'adamw_decay': [], 'adamw_no_decay': [], 'muon': []}
    names = {name: [] for name in parameters}
    aliases = {}
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        aliases.setdefault(identity, []).append(name)
        if identity in seen:
            continue
        seen.add(identity)
        if identity in hidden and parameter.ndim == 2:
            group = 'muon'
        elif identity in excluded or parameter.ndim < 2 or name.endswith('.bias'):
            group = 'adamw_no_decay'
        else:
            group = 'adamw_decay'
        parameters[group].append(parameter)
        names[group].append(name)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected or sum(map(len, parameters.values())) != len(expected):
        raise ValueError('Every trainable parameter must be assigned exactly once')
    if not parameters['muon'] or not parameters['adamw_no_decay']:
        raise ValueError('Muon requires hidden matrices and auxiliary AdamW parameters')
    groups = [dict(params=parameters[name], name=name, lr=cfg['learning_rate'],
                   betas=tuple(cfg['betas']), eps=cfg['epsilon'],
                   weight_decay=cfg['weight_decay'] if name == 'adamw_decay' else 0.0)
              for name in ('adamw_decay', 'adamw_no_decay')]
    muon = cfg['muon']
    groups.append(dict(params=parameters['muon'], name='muon',
                       lr=muon['learning_rate'], momentum=muon['momentum'],
                       nesterov=muon['nesterov'], ns_steps=muon['ns_steps'],
                       weight_decay=muon['weight_decay'], adjust_lr_fn=muon['adjust_lr_fn'], eps=muon['epsilon'],
                       ns_coefficients=tuple(muon['ns_coefficients'])))
    metadata = dict(primary_learning_rate_group='adamw_no_decay', parameter_names=names,
                    parameter_counts={name: sum(p.numel() for p in values)
                                      for name, values in parameters.items()},
                    tied_aliases=[value for value in aliases.values() if len(value) > 1],
                    muon_configuration=muon, implementation='torch.optim.Muon',
                    torch_version=torch.__version__,
                    distribution='replicated DDP, complete identical optimizer state on each rank',
                    state_dtype='native PyTorch optimizer states; BF16 autocast training',
                    scheduler='same cosine/warmup multiplier on AdamW and Muon base learning rates',
                    weight_decay='decoupled using each group learning rate before its update')
    return groups, metadata


def build_muon_optimizers(model, cfg):
    if not hasattr(torch.optim, 'Muon'):
        raise RuntimeError('Native torch.optim.Muon requires the isolated PyTorch 2.9.1 environment')
    groups, metadata = muon_groups(model, cfg)
    adamw = torch.optim.AdamW(groups[:2], lr=cfg['learning_rate'],
                             betas=tuple(cfg['betas']), eps=cfg['epsilon'], fused=True)
    muon = torch.optim.Muon(groups[2:])
    return [adamw, muon], metadata
