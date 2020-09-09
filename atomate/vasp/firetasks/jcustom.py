from fireworks import FiretaskBase, explicit_serialize
from pymatgen.io.vasp.inputs import Structure, Poscar
from pymatgen.io.vasp.sets import MPStaticSet
from atomate.common.firetasks.glue_tasks import get_calc_loc, PassResult, \
    CopyFiles, CopyFilesFromCalcLoc
import shutil
import gzip
import os
import re

@explicit_serialize
class RmSelectiveDynPoscar(FiretaskBase):
    def run_task(self, fw_spec):
        input_strucutre = Structure.from_file("POSCAR")
        if "selective_dynamics" in input_strucutre.site_properties.keys():
            input_strucutre.remove_site_property("selective_dynamics")
            input_strucutre.to("POSCAR", "POSCAR")

@explicit_serialize
class SelectiveDynmaicPoscar(FiretaskBase):
    _fw_name = "SelectiveDynamicPoscar"
    required_params = ["selective_dynamics", "nsites"]

    def run_task(self, fw_spec):
        where = []
        for i in range(self["nsites"]):
            if i in self["selective_dynamics"]:
                where.append([True, True, True])
            else:
                where.append([False, False, False])
        poscar_selective = Poscar.from_file("POSCAR")
        poscar_selective.selective_dynamics = where
        poscar_selective.write_file("POSCAR")


@explicit_serialize
class JWriteScanVaspStaticFromPrev(FiretaskBase):
    """
    Writes input files for a static run. Assumes that output files from a
    previous (e.g., optimization) run can be accessed in current dir or
    prev_calc_dir. Also allows lepsilon (dielectric constant) calcs.

    Optional params:
        potcar_spec (bool): Instead of writing the POTCAR, write a
            "POTCAR.spec". This is intended to allow testing of workflows
            without requiring pseudo-potentials to be installed on the system.
        (documentation for all other optional params can be found in
        MPStaticSet)

    """

    optional_params = [
        "prev_calc_dir",
        "reciprocal_density",
        "small_gap_multiply",
        "standardize",
        "sym_prec",
        "international_monoclinic",
        "lepsilon",
        "other_params",
        "potcar_spec",
    ]

    def run_task(self, fw_spec):
        lepsilon = self.get("lepsilon")

        # more k-points for dielectric calc.
        default_reciprocal_density = 200 if lepsilon else 100
        other_params = self.get("other_params", {})
        user_incar_settings = other_params.get("user_incar_settings", {})

        # for lepsilon runs, set EDIFF to 1E-5 unless user says otherwise
        if (
                lepsilon
                and "EDIFF" not in user_incar_settings
                and "EDIFF_PER_ATOM" not in user_incar_settings
        ):
            if "user_incar_settings" not in other_params:
                other_params["user_incar_settings"] = {}

        updates = {
            "ADDGRID": True,
            "LASPH": True,
            "LDAU": False,
            "LMIXTAU": True,
            "METAGGA": "SCAN",
            "NELM": 200
        }
        print(other_params)
        other_params["user_incar_settings"].update(updates)
        print("o!"*50)
        vis = MPStaticSet.from_prev_calc(
            prev_calc_dir=self.get("prev_calc_dir", "."),
            reciprocal_density=self.get(
                "reciprocal_density", default_reciprocal_density
            ),
            small_gap_multiply=self.get("small_gap_multiply", None),
            standardize=self.get("standardize", False),
            sym_prec=self.get("sym_prec", 0.1),
            international_monoclinic=self.get(
                "international_monoclinic", True
            ),
            lepsilon=lepsilon,
            **other_params
        )

        potcar_spec = self.get("potcar_spec", False)
        vis.write_input(".", potcar_spec=potcar_spec)
