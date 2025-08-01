# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['build_head']


def build_head(config, **kwargs):
    # det head
    from .det_db_head import DBHead, PFHeadLocal
    from .det_east_head import EASTHead
    from .det_sast_head import SASTHead
    from .det_pse_head import PSEHead
    from .det_fce_head import FCEHead
    from .e2e_pg_head import PGHead

    # rec head
    from .rec_ctc_head import CTCHead
    from .rec_att_head import AttentionHead
    from .rec_srn_head import SRNHead
    from .rec_nrtr_head import Transformer
    from .rec_sar_head import SARHead
    from .rec_can_head import CANHead
    from .rec_multi_head import MultiHead

    # cls head
    from .cls_head import ClsHead
    support_dict = [
        'DBHead', 'PSEHead', 'EASTHead', 'SASTHead', 'CTCHead', 'ClsHead', 'AttentionHead',
        'SRNHead', 'PGHead', 'Transformer', 'TableAttentionHead','SARHead', 'FCEHead',
        'CANHead', 'MultiHead', 'PFHeadLocal',

    ]

    from .table_att_head import TableAttentionHead

    module_name = config.pop('name')
    assert module_name in support_dict, Exception('head only support {}'.format(
        support_dict))
    module_class = eval(module_name)(**config, **kwargs)
    return module_class