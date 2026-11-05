"""Issue, pending readiness, RAW overlap and synchronization."""
from collections import Counter

from measurement.qualify import require


class Timeline:

    def __init__(self):
        self.clock = 0.
        self.pending = {}
        self.trace = []

    def advance(self, ticks):
        self.clock += ticks
        self.pending = {k: max(v - ticks, 0.) for k, v in self.pending.items() if v > ticks}

    def drain(self, producers=None):
        keys = set(self.pending) if producers is None else set(producers)
        wait = max((self.pending.get(p, 0.) for p in keys), default=0.)
        self.advance(wait)
        return wait

    def issue(self, event, independent, dependent):
        wait = self.drain(event['deps'])
        issue = event['count'] * independent
        self.advance(issue)
        tail = max(dependent - independent, 0.)
        if tail: self.pending[event['id']] = tail
        self.trace.append({
            'id': event['id'], 'kind': event['kind'], 'native': event['native'], 'count': event['count'], 'raw_wait':
            wait, 'issue': issue, 'pending': tail, 'clock': self.clock
        })


def predict(events, rates, memory_d):
    sim = Timeline()
    sources = Counter()
    coverage = Counter()
    formal_producers = {}

    def cost(kind, native, event=None):
        key = native if native in rates else kind
        require(key in rates, 'primitive cost missing: ' + str(native))
        row = rates[key]
        sources[row['source']] += 1
        if 'measurements' in row:
            matches = [
                m for m in row['measurements'] if m['conditions'] == event.get('conditions')
                and tuple(m['input_restrictions']) == tuple(event.get('input_restrictions', ())) and (
                    m['metric'] != 'dependent_latency' or m['raw_role'] == event.get('raw_role')) and (
                        m['metric'] != 'independent_issue' or event['count'] <= m['independent_chains'])
            ]
            require(any(m['metric'] == 'independent_issue' for m in matches),
                    'model gap: independent cost scope not proven for ' + str(native))
            require(any(m['metric'] == 'dependent_latency' for m in matches),
                    'model gap: dependent cost scope not proven for ' + str(native))
            # Equal native opcodes can have distinct selectors or operand bindings.
            independent = next(m['value'] for m in matches if m['metric'] == 'independent_issue')
            dependent = next(m['value'] for m in matches if m['metric'] == 'dependent_latency')
            formal_producers[event['id']] = dict(native=native, raw_role=event['raw_role'],
                                                 conditions=event['conditions'])
            return independent, dependent, dict(status='formal', measurements=[m['measurement_id'] for m in matches])
        d = row.get('D')
        if d is None:
            axis = 'lds_d' if kind == 'shared_load' else 'sts_d' if kind == 'shared_store' else 'gvm_d'
            require(kind in ('shared_load', 'shared_store', 'global_load', 'global_store'), 'unmeasured non-memory D')
            d = memory_d[axis]
        return row['I'], d, dict(status='provisional', reason=row['source'])

    for event in events:
        if event['kind'] == 'drain':
            require(not any(p in formal_producers for p in sim.pending),
                    'model gap: same-op RAW latency does not prove arbitrary sync completion')
            wait = sim.drain()
            sim.trace.append({'id': event['id'], 'kind': 'drain', 'wait': wait, 'clock': sim.clock})
        elif event['kind'] == 'rearrange':
            before = sim.clock
            require(not any(p in formal_producers for p in sim.pending),
                    'model gap: unmeasured rearrange consumer of formal RAW latency')
            drain = sim.drain()
            si, sd, store_scope = cost('shared_store', event['store_native'])
            li, ld, load_scope = cost('shared_load', event['load_native'])
            sim.advance(event['stores'] * si + max(sd - si, 0.))
            sim.advance(event['loads'] * li)
            tail = max(ld - li, 0.)
            if tail: sim.pending[event['id']] = tail
            sim.trace.append({
                'id': event['id'], 'kind': 'rearrange', 'drain': drain, 'stores': event['stores'], 'loads':
                event['loads'], 'cost': sim.clock - before, 'pending': tail, 'clock': sim.clock
            })
            coverage[store_scope['status']] += event['stores']
            coverage[load_scope['status']] += event['loads']
            sim.trace[-1]['cost_scope'] = dict(store=store_scope, load=load_scope)
        else:
            for producer in event['deps']:
                if producer in sim.pending and producer in formal_producers:
                    observed = formal_producers[producer]
                    require(
                        observed['native'] == event['native'] and observed['conditions'] == event.get('conditions')
                        and producer in event.get('operand_deps', {}).get(observed['raw_role'], ()),
                        'model gap: unmeasured producer-to-consumer RAW role')
            independent, dependent, scope = cost(event['kind'], event['native'], event)
            sim.issue(event, independent, dependent)
            coverage[scope['status']] += event['count']
            sim.trace[-1]['cost_scope'] = scope
            sim.trace[-1].update(
                {key: event[key]
                 for key in ('operand_uses', 'operand_deps', 'memory')
                 if key in event})
    return {
        'predicted_cost': sim.clock, 'trace': sim.trace, 'pending_at_return': sim.pending, 'sources': dict(sources),
        'coverage': dict(formal=coverage['formal'], provisional=coverage['provisional'], unsupported=0)
    }
