from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen.io.vasp.inputs import Structure, Kpoints, Poscar

from atomate.vasp.fireworks.core import OptimizeFW, StaticFW, ScanOptimizeFW
from atomate.vasp.fireworks.jcustom import *
from atomate.vasp.powerups import use_fake_vasp, add_namefile, add_additional_fields_to_taskdocs, preserve_fworker, \
    add_modify_incar, add_modify_kpoints, set_queue_options, set_execution_options

from fireworks import Firework, LaunchPad, Workflow

import numpy as np


def get_wf_full_hse(structure, charge_states, gamma_only, dos, nupdowns, task, catagory, encut=520,
                    vasptodb=None, wf_addition_name=None):

    vasptodb = vasptodb or {}

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
            "SIGMA": 0.001
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


        # FW1 Structure optimization firework
        opt = JOptimizeFW(
            structure=structure,
            name="PBE_relax",
            max_force_threshold=False,
            job_type="normal",
            override_default_vasp_params={
                "user_incar_settings": user_incar_settings,
                "user_kpoints_settings": user_kpoints_settings
            },
        )

        # FW2 Run HSE relax
        def hse_relax(parents):
            fw = JHSERelaxFW(
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
                parents=parents
            )
            return fw

        # FW3 Run HSE SCF
        uis_hse_scf = {
            "user_incar_settings": {
                "LVHAR": True,
                # "AMIX": 0.2,
                # "AMIX_MAG": 0.8,
                # "BMIX": 0.0001,
                # "BMIX_MAG": 0.0001,
                "EDIFF": 1.0e-07,
                "ENCUT": encut,
                "ISMEAR": 0,
                "LCHARG": False,
                "NSW": 0,
                "NUPDOWN": nupdown,
                "NELM": 150,
                "SIGMA": 0.05
            },
            "user_kpoints_settings": user_kpoints_settings
        }

        if dos:
            uis_hse_scf["user_incar_settings"].update({"ENMAX": 10, "ENMIN": -10, "NEDOS": 9000})

        uis_hse_scf["user_incar_settings"].update({"NELECT": nelect})

        def hse_scf(parents):
            fw = JHSEStaticFW(
                structure,
                vasp_input_set_params=uis_hse_scf,
                parents=parents,
                name="HSE_scf",
                vasptodb_kwargs={
                    "additional_fields": {
                        "task_type": "JHSEStaticFW",
                        "charge_state": cs,
                        "nupdown_set": nupdown
                    }
                }
            )
            return fw

        if task == "opt":
            fws.append(opt)
        elif task == "hse_relax":
            fws.append(hse_relax(parents=None))
        elif task == "hse_scf":
            fws.append(hse_scf(parents=None))
        elif task == "hse_relax-hse_scf":
            fws.append(hse_relax(parents=None))
            fws.append(hse_scf(fws[-1]))
        elif task == "opt-hse_relax-hse_scf":
            fws.append(opt)
            fws.append(hse_relax(parents=fws[-1]))
            fws.append(hse_scf(parents=fws[-1]))

    wf_name = "{}:{}:q{}:sp{}".format("".join(structure.formula.split(" ")), wf_addition_name, charge_states, nupdowns)

    wf = Workflow(fws, name=wf_name)

    vasptodb.update({"wf": [fw.name for fw in wf.fws]})
    wf = add_additional_fields_to_taskdocs(wf, vasptodb)
    wf = add_modify_incar(wf)
    wf = preserve_fworker(wf)
    wf = add_namefile(wf)
    wf = set_execution_options(wf, category=catagory)
    return wf


def get_wf_full_scan(structure, charge_states, gamma_only, dos, nupdowns, task, catagory, encut=520,
                     vasptodb=None, wf_addition_name=None):

    vasptodb = vasptodb or {}

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
            "SIGMA": 0.001,
            "EDIFFG": -0.01,
            "LCHARG": True,
            "NUPDOWN": nupdown,
        }

        user_incar_settings.update({"NELECT": nelect})

        if gamma_only is True:
            user_kpoints_settings = Kpoints.gamma_automatic()
            # user_kpoints_settings = MPRelaxSet(structure).kpoints

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
        scan_relax = JScanOptimizeFW(
            structure=structure,
            override_default_vasp_params={
                "user_incar_settings": user_incar_settings,
                "user_kpoints_settings": user_kpoints_settings
            },
            job_type="normal",
            max_force_threshold=False,
            name="SCAN_relax",
            vasptodb_kwargs={
                "additional_fields": {
                    "task_type": "JScanOptimizeFW",
                    "charge_state": cs,
                    "nupdown_set": nupdown,
                },
                "parse_dos": True,
                "parse_eigenvalues": True,
                "parse_chgcar": True
            }
        )

        # FW2 Run SCAN SCF
        uis_scan_scf = {
            "user_incar_settings": {
                "LAECHG": False,
                "EDIFF": 1e-05,
                "ENCUT": encut,
                "ISMEAR": 0,
                "LCHARG": True,
                "NUPDOWN": nupdown,
            },
            "user_kpoints_settings": user_kpoints_settings
        }

        if dos:
            uis_scan_scf["user_incar_settings"].update({"EMAX": 10, "EMIN": -10, "NEDOS": 9000})

        uis_scan_scf["user_incar_settings"].update({"NELECT": nelect})

        def scan_scf(parents):
            fw = JScanStaticFW(
                structure=structure,
                vasp_input_set_params=uis_scan_scf,
                parents=parents,
                name="SCAN_scf",
                vasptodb_kwargs={
                    "additional_fields": {
                        "task_type": " JScanStaticFW",
                        "charge_state": cs,
                        "nupdown_set": nupdown,
                    },
                    "parse_dos": True,
                    "parse_chgcar": True
                })
            return fw

        if task == "scan_relax":
            fws.append(scan_relax)
        elif task == "scan_scf":
            fws.append(scan_scf(None))
        elif task == "scan_relax-scan_scf":
            fws.append(scan_relax)
            fws.append(scan_scf(fws[-1]))

    wf_name = "{}:{}:q{}:sp{}".format("".join(structure.formula.split(" ")), wf_addition_name, charge_states, nupdowns)
    wf = Workflow(fws, name=wf_name)
    vasptodb.update({"wf": [fw.name for fw in wf.fws]})
    wf = add_additional_fields_to_taskdocs(wf, vasptodb)
    wf = set_execution_options(wf, category=catagory)
    wf = preserve_fworker(wf)
    wf = add_namefile(wf)
    wf = add_modify_incar(wf)
    return wf