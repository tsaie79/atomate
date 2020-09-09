from fireworks import FiretaskBase, explicit_serialize
from pymatgen.io.vasp.inputs import Structure, Poscar
from pymatgen.io.vasp.sets import MPScanStaticSet
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
        "has_KPOINTS"
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
            other_params["user_incar_settings"]["EDIFF"] = 1e-5
            other_params["user_incar_settings"]["LMIXTAU"] = True
            other_params["user_incar_settings"]["LREAL"] = "Auto"

            if self.get("has_KPOINTS", True):
                other_params["user_incar_settings"].pop("KSPACING")

        vis = MPScanStaticSet.from_prev_calc(
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


@explicit_serialize
class JCopyScanVaspOutputs(CopyFiles):
    """
    Copy files from a previous VASP run directory to the current directory.
    By default, copies 'INCAR', 'POSCAR' (default: via 'CONTCAR'), 'KPOINTS',
    'POTCAR', 'OUTCAR', and 'vasprun.xml'. Additional files, e.g. 'CHGCAR',
    can also be specified. Automatically handles files that have a ".gz"
    extension (copies and unzips).

    Note that you must specify either "calc_loc" or "calc_dir" to indicate
    the directory containing the previous VASP run.

    Required params:
        (none) - but you must specify either "calc_loc" OR "calc_dir"

    Optional params:
        calc_loc (str OR bool): if True will set most recent calc_loc. If str
            search for the most recent calc_loc with the matching name
        calc_dir (str): path to dir that contains VASP output files.
        filesystem (str): remote filesystem. e.g. username@host
        additional_files ([str]): additional files to copy,
            e.g. ["CHGCAR", "WAVECAR"]. Use $ALL if you just want to copy
            everything
        contcar_to_poscar(bool): If True (default), will move CONTCAR to
            POSCAR (original POSCAR is not copied).
    """

    optional_params = ["calc_loc", "calc_dir", "filesystem", "additional_files",
                       "contcar_to_poscar", "has_KPOINTS"]

    def run_task(self, fw_spec):

        calc_loc = get_calc_loc(self["calc_loc"],
                                fw_spec["calc_locs"]) if self.get(
            "calc_loc") else {}

        # determine what files need to be copied
        files_to_copy = None
        if not "$ALL" in self.get("additional_files", []):
            files_to_copy = ['INCAR', 'POSCAR', 'KPOINTS', 'POTCAR', 'OUTCAR',
                             'vasprun.xml']
            if not self.get("has_KPOINTS", False):
                files_to_copy.remove("KPOINTS")
            if self.get("additional_files"):
                files_to_copy.extend(self["additional_files"])

        # decide between poscar and contcar
        contcar_to_poscar = self.get("contcar_to_poscar", True)
        if contcar_to_poscar and "CONTCAR" not in files_to_copy:
            files_to_copy.append("CONTCAR")
            files_to_copy = [f for f in files_to_copy if
                             f != 'POSCAR']  # remove POSCAR

        # setup the copy
        self.setup_copy(self.get("calc_dir", None),
                        filesystem=self.get("filesystem", None),
                        files_to_copy=files_to_copy, from_path_dict=calc_loc)
        # do the copying
        self.copy_files()

    def copy_files(self):
        all_files = self.fileclient.listdir(self.from_dir)
        # start file copy
        for f in self.files_to_copy:
            prev_path_full = os.path.join(self.from_dir, f)
            dest_fname = 'POSCAR' if f == 'CONTCAR' and self.get(
                "contcar_to_poscar", True) else f
            dest_path = os.path.join(self.to_dir, dest_fname)

            relax_ext = ""
            relax_paths = sorted(
                self.fileclient.glob(prev_path_full + ".relax*"))
            if relax_paths:
                if len(relax_paths) > 9:
                    raise ValueError(
                        "CopyVaspOutputs doesn't properly handle >9 relaxations!")
                m = re.search('\.relax\d*', relax_paths[-1])
                relax_ext = m.group(0)

            # detect .gz extension if needed - note that monty zpath() did not seem useful here
            gz_ext = ""
            if not (f + relax_ext) in all_files:
                for possible_ext in [".gz", ".GZ"]:
                    if (f + relax_ext + possible_ext) in all_files:
                        gz_ext = possible_ext

            if not (f + relax_ext + gz_ext) in all_files:
                raise ValueError("Cannot find file: {}".format(f))

            # copy the file (minus the relaxation extension)
            self.fileclient.copy(prev_path_full + relax_ext + gz_ext,
                                 dest_path + gz_ext)

            # unzip the .gz if needed
            if gz_ext in ['.gz', ".GZ"]:
                # unzip dest file
                with open(dest_path, 'wb') as f_out, gzip.open(dest_path + gz_ext, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(dest_path + gz_ext)