from pymatgen.io.vasp.sets import MPRelaxSet, MVLGWSet
from pymatgen.io.vasp.inputs import Structure, Kpoints, Poscar

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, ScanOptimizeFW
from atomate.vasp.fireworks.jcustom import *
from atomate.vasp.powerups import use_fake_vasp, add_namefile, add_additional_fields_to_taskdocs, preserve_fworker, \
    add_modify_incar, add_modify_kpoints, set_queue_options, set_execution_options

from fireworks import Firework, LaunchPad, Workflow

import numpy as np


def gw_wf(structure, ncores, vis_static=None, vasp_input_set_params=None, vasptodb=None, wf_addition_name=None):
    fws = []
    # 1. STATIC
    static_fw = StaticFW(structure, vasp_input_set=vis_static, vasp_input_set_params={"force_gamma": True},
                         name="gw_static") #ediff=1e-4

    # 2. DIAG
    diag_fw = JMVLGWFW(structure, ncores=ncores, parents=static_fw, vasp_input_set_params=vasp_input_set_params,
                       mode="DIAG", name="gw_diag")

    # 3. GW
    gw_fw = JMVLGWFW(structure, ncores=ncores, parents=diag_fw, vasp_input_set_params=vasp_input_set_params,
                     mode="GW", name="gw_gw")

    # 4. BSE
    bse_fw = JMVLGWFW(structure, ncores=ncores, parents=gw_fw, vasp_input_set_params=vasp_input_set_params,
                      mode="BSE", name="gw_bse")

    fws.append(static_fw)
    fws.append(diag_fw)
    fws.append(gw_fw)
    fws.append(bse_fw)

    wf_name = "{}:{}".format("".join(structure.formula.split(" ")), wf_addition_name)
    wf = Workflow(fws, name=wf_name)
    vasptodb.update({"wf": [fw.name for fw in wf.fws]})
    wf = add_additional_fields_to_taskdocs(wf, vasptodb)
    wf = add_modify_incar(wf, {"incar_update": {"MAGMOM": MPRelaxSet(structure).incar.get("MAGMOM", None)}})
    wf = add_namefile(wf)
    wf = add_modify_incar(wf)
    return wf

