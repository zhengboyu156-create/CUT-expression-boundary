"""CUT random target B plus a separate, geometrically synchronized real reference."""
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch
from data.base_dataset import BaseDataset
from expression.manifest import read_manifest
from expression.statistics import foreground, to_tensor
from expression.protocol import policy_hash


class ExpressionPairDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--pair_manifest', required=True)
        parser.add_argument('--pair_split', default='train')
        parser.add_argument('--demo_overfit', action='store_true')
        parser.add_argument('--image_cache_size',type=int,default=16)
        parser.set_defaults(preprocess='none')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if opt.direction != 'AtoB' or opt.input_nc != 3 or opt.output_nc != 3:
            raise ValueError('Expression proxies require RGB H&E to RGB IHC.')
        if opt.preprocess != 'none':
            raise ValueError('Use preprocess=none; geometry is defined by the shared pixel window.')
        if opt.pair_split == 'demo' and opt.isTrain and not opt.demo_overfit:
            raise ValueError('Training on the attachment requires explicit --demo_overfit.')
        if opt.isTrain and opt.pair_split not in ('train', 'demo'):
            raise ValueError('G training is restricted to train, or explicitly marked demo.')
        self.rows = [r for r in read_manifest(opt.pair_manifest) if r['split']==opt.pair_split and r['valid']]
        self._images=OrderedDict();self.cache_hits=0;self.cache_misses=0
        self.cache_limit=getattr(opt,'image_cache_size',16)
        if self.cache_limit<0:raise ValueError('Image cache size must be nonnegative.')
        if not self.rows: raise ValueError('No valid windows in requested split.')
        if opt.pair_split!='demo' and any(r.get('protocol_hash')!=policy_hash(opt.stats_window) for r in self.rows):
            raise ValueError('Formal data must be prepared with the current sampling/quality protocol.')
        if any(r['he_bbox'][2:] != [opt.stats_window, opt.stats_window] for r in self.rows):
            raise ValueError('Manifest window and model stats_window differ.')

    def image(self, path):
        if path in self._images:
            self.cache_hits+=1;self._images.move_to_end(path);return self._images[path]
        self.cache_misses+=1
        rgb = np.array(Image.open(path).convert('RGB'))
        result=(rgb,foreground(rgb))
        if self.cache_limit:
            self._images[path]=result
            while len(self._images)>self.cache_limit:self._images.popitem(last=False)
        return result

    def crop(self, row, key):
        rgb, mask = self.image(row[key]); x,y,w,h = row[key+'_bbox']
        return to_tensor(rgb[y:y+h,x:x+w]), torch.from_numpy(mask[y:y+h,x:x+w].copy()[None]).float()

    def __getitem__(self, index):
        row = self.rows[index]
        a, mask = self.crop(row, 'he'); ref, _ = self.crop(row, 'ihc')
        # The independent target draw cannot change the paired reference.
        j = index if self.opt.serial_batches else int(torch.randint(len(self.rows), ()).item())
        b, _ = self.crop(self.rows[j], 'ihc')
        if self.opt.isTrain and not self.opt.no_flip and bool(torch.rand(())>.5):
            a, ref, mask = [torch.flip(t, [-1]) for t in (a, ref, mask)]
        return dict(A=a, B=b, paired_B=ref, tissue_mask=mask,
                    A_paths=row['he'], B_paths=self.rows[j]['ihc'],
                    window_id=row['window_id'], group_id=row['group_id'],
                    bbox=torch.tensor(row['he_bbox']), paired_bbox=torch.tensor(row['ihc_bbox']))

    def __len__(self):
        return len(self.rows)
