"""H&E-to-H&E retrieval; each window keeps its own same-coordinate IHC."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def fit_morphology(rows, demo=False):
    rows=[r for r in rows if r['valid']]
    allowed={'demo'} if demo else {'train'}
    if not rows or any(r['split'] not in allowed for r in rows):
        raise ValueError('Morphology scaler and clusters must be fitted on train only.')
    if len(rows)<4: raise ValueError('Need at least four valid morphology windows.')
    x=np.array([r['features'] for r in rows]);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6)
    z=(x-mean)/scale; best=None
    for k in range(2,min(5,len(rows)-1)+1):
        model=KMeans(n_clusters=k,n_init=10,random_state=0).fit(z)
        if len(np.unique(model.labels_))<2:continue
        score=float(silhouette_score(z,model.labels_))
        if best is None or score>best[0]:best=(score,model.cluster_centers_)
    centers=z.mean(0,keepdims=True) if best is None else best[1]
    return dict(mean=mean.tolist(),scale=scale.tolist(),centers=centers.tolist(),
                silhouette=None if best is None else best[0],
                fit_groups=sorted({r['group_id'] for r in rows}),demo_only=demo,
                distance_threshold=1.5,delta_threshold=.20,measurement='fixed_HE_mask_DAB_area')


def retrieve(rows,state,demo=False,neighbors=16,query_batch_size=64):
    """Enumerate all eligible H&E neighbours inside the frozen radius.

    ``neighbors`` is retained for callers of the former API. It no longer caps
    the candidate set: a top-k cap silently missed valid expression conflicts
    in dense morphology clusters. No IHC location is searched or changed.
    """
    rows=[r for r in rows if r['valid']]
    if not demo:
        if state['demo_only'] or any(r['split']!='test' for r in rows):
            raise ValueError('Formal challenge retrieval requires held-out test and a train-fitted representation.')
        if set(state['fit_groups']) & {r['group_id'] for r in rows}:
            raise ValueError('Morphology fitting groups overlap held-out challenge groups.')
    if len(rows)<2:return []
    z=(np.array([r['features'] for r in rows])-state['mean'])/state['scale']
    centers=np.array(state['centers']); labels=((z[:,None]-centers)**2).sum(2).argmin(1)
    if query_batch_size<1:raise ValueError('Query batch size must be positive.')
    nn=NearestNeighbors(radius=state['distance_threshold']).fit(z)
    pairs=[]
    # Bound temporary neighbour arrays; do not retain a set of every nearby
    # non-conflicting pair. Symmetric radius queries allow j > i deduplication.
    for start in range(0,len(rows),query_batch_size):
        distances,indices=nn.radius_neighbors(z[start:start+query_batch_size],sort_results=True)
        for local,(ds,js) in enumerate(zip(distances,indices)):
            i=start+local
            for distance,j in zip(ds,js):
                j=int(j)
                if j<=i or labels[i]!=labels[j]:continue
                a,b=rows[i],rows[j]
                if not demo and a['group_id']==b['group_id']:continue
                delta=abs(a['q']-b['q'])
                if distance<=state['distance_threshold'] and delta>=state['delta_threshold']:
                    pairs.append(dict(a=a['window_id'],b=b['window_id'],distance=float(distance),delta=delta,
                        he_a=a['he_bbox'],ihc_a=a['ihc_bbox'],he_b=b['he_bbox'],ihc_b=b['ihc_bbox'],
                        scope='within-source proxy demo' if demo else 'held-out cross-group staining-proxy challenge'))
    return sorted(pairs,key=lambda p:p['distance'])
