from pymatgen.io.vasp.sets import MPScanStaticSet
from pymatgen.io.vasp.inputs import Structure, Kpoints, Poscar

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, ScanOptimizeFW
from atomate.vasp.fireworks.jcustom import *
from atomate.vasp.powerups import use_fake_vasp, add_namefile, add_additional_fields_to_taskdocs, preserve_fworker, \
    add_modify_incar, add_modify_kpoints, set_queue_options, set_execution_options

from fireworks import Firework, LaunchPad, Workflow

import numpy as np


def get_wf_full_hse(structure, charge_states, gamma_only, dos_hse, nupdowns, encut=520,
                    include_hse_relax=False, vasptodb=None, wf_addition_name=None):
    fws = []
    for cs, nupdown in zip(charge_states, nupdowns):
        print("Formula: {}".format(structure.formula))
        if structure.site_properties.get("magmom", None):
            structure.remove_site_property("magmom")
        structure.set_charge(cs)
        nelect = MPRelaxSet(structure, use_structure_charge=True).nelect
        user_incar_settings = {
            "ENCUT": encut,
            "ISIF": 2,
            "ISMEAR": 0,
            "EDIFFG": -0.01,
            "LCHARG": False,
            "NUPDOWN": nupdown,
            #"NCORE": 4 owls normal 14; cori 8. Reduce ncore if want to increase speed but low memory risk
        }

        user_incar_settings.update({"NELECT": nelect})

        if gamma_only is True:
            # user_kpoints_settings = Kpoints.gamma_automatic((1,1,1), (0.333, 0.333, 0))
            user_kpoints_settings = Kpoints.gamma_automatic()

        elif gamma_only:
            nkpoints = len(gamma_only)
            kpts_weights = [1.0 for i in np.arange(nkpoints)]
            labels = [None for i in np.arange(nkpoints)]
            user_kpoints_settings = Kpoints.from_dict(
                {
                    'comment': 'JCustom',
                    'nkpoints': nkpoints,
                    'generation_style': 'Reciprocal',
                    'kpoints': gamma_only,
                    'usershift': (0, 0, 0),
                    'kpts_weights': kpts_weights,
                    'coord_type': None,
                    'labels': labels,
                    'tet_number': 0,
                    'tet_weight': 0,
                    'tet_connections': None,
                    '@module': 'pymatgen.io.vasp.inputs',
                    '@class': 'Kpoints'
                }
            )

        else:
            user_kpoints_settings = None

        # input set for relaxation
        vis_relax = MPRelaxSet(structure, force_gamma=True)
        v = vis_relax.as_dict()
        v.update({"user_incar_settings": user_incar_settings, "user_kpoints_settings": user_kpoints_settings})
        vis_relax = vis_relax.__class__.from_dict(v)

        # FW1 Structure optimization firework
        opt = OptimizeFW(
            structure=structure,
            vasp_input_set=vis_relax,
            db_file=DB_FILE if DB_FILE else '>>db_file<<',
            name="PBE_relax",
            max_force_threshold=False,
            job_type="normal"
        )

        # FW2 Run HSE relax
        hse_relax = HSERelaxFW(
            structure=structure,
            vasp_input_set_params={
                "user_incar_settings": user_incar_settings,
                "user_kpoints_settings": user_kpoints_settings
            },
            name="HSE_relax",
            vasptodb_kwargs={"additional_fields": {
                "charge_state": cs,
                "nupdown_set": nupdown
            }},
            parents=opt
        )

        # FW3 Run HSE SCF
        uis_hse_scf = {
            "user_incar_settings": {
                "LVHAR": True,
                # "AMIX": 0.2,
                # "AMIX_MAG": 0.8,
                # "BMIX": 0.0001,
                # "BMIX_MAG": 0.0001,
                "EDIFF": 1.0e-05,
                "ENCUT": encut,
                "ISMEAR": 0,
                "LCHARG": False,
                "NSW": 0,
                "NUPDOWN": nupdown,
                "NELM": 150
            },
            "user_kpoints_settings": user_kpoints_settings
        }

        if dos_hse:
            uis_hse_scf["user_incar_settings"].update({"ENMAX": 10, "ENMIN": -10, "NEDOS": 9000})

        uis_hse_scf["user_incar_settings"].update({"NELECT": nelect})

        if include_hse_relax:
            parent_hse_scf = hse_relax
        else:
            parent_hse_scf = opt
        scf = HSEStaticFW(structure,
                          vasp_input_set_params=uis_hse_scf,
                          parents=parent_hse_scf,
                          name="HSE_scf",
                          vasptodb_kwargs={"additional_fields": {
                              "task_type": "HSEStaticFW",
                              "charge_state": cs,
                              "nupdown_set": nupdown
                          }})
        fws.append(opt)
        if include_hse_relax:
            fws.append(hse_relax)
        fws.append(scf)

    wf_name = "{}:{}:q{}:sp{}".format("".join(structure.formula.split(" ")), wf_addition_name, charge_states, nupdowns)
    wf = Workflow(fws, name=wf_name)
    vasptodb.update({"wf": [fw.name for fw in wf.fws]})
    wf = add_additional_fields_to_taskdocs(wf, vasptodb)
    wf = add_namefile(wf)
    wf = add_modify_incar(wf)
    return wf


def get_wf_full_scan(structure, charge_states, gamma_only, dos_hse, nupdowns, encut=520,
                     vasptodb=None, wf_addition_name=None):
    fws = []
    for cs, nupdown in zip(charge_states, nupdowns):
        print("Formula: {}".format(structure.formula))
        if structure.site_properties.get("magmom", None):
            structure.remove_site_property("magmom")
        structure.set_charge(cs)
        nelect = MPRelaxSet(structure, use_structure_charge=True).nelect
        user_incar_settings = {
            "ENCUT": encut,
            "ISIF": 2,
            # "ISMEAR": 0,
            "EDIFFG": -0.01,
            "LCHARG": False,
            "NUPDOWN": nupdown,
        }

        user_incar_settings.update({"NELECT": nelect})

        if gamma_only is True:
            # user_kpoints_settings = Kpoints.gamma_automatic((1,1,1), (0.333, 0.333, 0))
            user_kpoints_settings = Kpoints.gamma_automatic()

        elif gamma_only:
            nkpoints = len(gamma_only)
            kpts_weights = [1.0 for i in np.arange(nkpoints)]
            labels = [None for i in np.arange(nkpoints)]
            user_kpoints_settings = Kpoints.from_dict(
                {
                    'comment': 'JCustom',
                    'nkpoints': nkpoints,
                    'generation_style': 'Reciprocal',
                    'kpoints': gamma_only,
                    'usershift': (0, 0, 0),
                    'kpts_weights': kpts_weights,
                    'coord_type': None,
                    'labels': labels,
                    'tet_number': 0,
                    'tet_weight': 0,
                    'tet_connections': None,
                    '@module': 'pymatgen.io.vasp.inputs',
                    '@class': 'Kpoints'
                }
            )

        else:
            user_kpoints_settings = None

        # FW1 Structure optimization firework
        scan_opt = ScanOptimizeFW(
            structure=structure,
            override_default_vasp_params={
                "user_incar_settings":user_incar_settings,
                "user_kpoints_settings":user_kpoints_settings
            },
            name="SCAN_relax"
        )

        # FW2 Run SCAN SCF
        scan_scf_vis = MPScanStaticSet(structure).incar.as_dict()
        uis_scan_scf = {
            "user_incar_settings": {
                # "AMIX": 0.2,
                # "AMIX_MAG": 0.8,
                # "BMIX": 0.0001,
                # "BMIX_MAG": 0.0001,
                "ENCUT": encut,
                # "ISMEAR": 0,
                "LCHARG": False,
                "NUPDOWN": nupdown,
            },
            "user_kpoints_settings": user_kpoints_settings
        }
        uis_scan_scf["user_incar_settings"].update(scan_scf_vis)

        if dos_hse:
            uis_scan_scf["user_incar_settings"].update({"ENMAX": 10, "ENMIN": -10, "NEDOS": 9000})

        uis_scan_scf["user_incar_settings"].update({"NELECT": nelect})

        scan_scf = StaticFW(
            structure=structure,
            vasp_input_set_params=uis_scan_scf,
            parents=scan_opt,
            name="SCAN_scf",
            vasptodb_kwargs={"additional_fields": {
                "task_type": "MPScanStaticSet",
                "charge_state": cs,
                "nupdown_set": nupdown
            }})
        fws.append(scan_opt)
        fws.append(scan_scf)

    wf_name = "{}:{}:q{}:sp{}".format("".join(structure.formula.split(" ")), wf_addition_name, charge_states, nupdowns)
    wf = Workflow(fws, name=wf_name)
    vasptodb.update({"wf": [fw.name for fw in wf.fws]})
    wf = add_additional_fields_to_taskdocs(wf, vasptodb)
    wf = add_modify_incar(wf, {"incar_update": {"MAGMOM": MPRelaxSet(structure).incar.get("MAGMOM", None)}})
    wf = add_namefile(wf)
    wf = add_modify_incar(wf)
    return wf