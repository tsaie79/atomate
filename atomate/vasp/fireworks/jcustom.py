import warnings

from fireworks import Firework

from pymatgen import Structure
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPHSERelaxSet,
    MPScanStaticSet,
    MPScanRelaxSet,
    MVLScanRelaxSet,
    MITMDSet,
    MITRelaxSet,
    MPStaticSet,
    MPSOCSet,
)

from atomate.common.firetasks.glue_tasks import (
    PassCalcLocs,
    GzipDir,
    CopyFiles,
    DeleteFiles,
    CopyFilesFromCalcLoc,
)
from atomate.vasp.config import (
    HALF_KPOINTS_FIRST_RELAX,
    RELAX_MAX_FORCE,
    VASP_CMD,
    DB_FILE,
    VDW_KERNEL_DIR,
)
from atomate.vasp.firetasks.glue_tasks import CopyVaspOutputs, pass_vasp_result
from atomate.vasp.firetasks.neb_tasks import TransferNEBTask
from atomate.vasp.firetasks.parse_outputs import VaspToDb, BoltztrapToDb
from atomate.vasp.firetasks.run_calc import (
    RunVaspCustodian,
    RunBoltztrap,
)
from atomate.vasp.firetasks.write_inputs import (
    WriteNormalmodeDisplacedPoscar,
    WriteTransmutedStructureIOSet,
    WriteVaspFromIOSet,
    WriteVaspHSEBSFromPrev,
    WriteVaspFromPMGObjects,
    WriteVaspNSCFFromPrev,
    WriteVaspSOCFromPrev,
    WriteVaspStaticFromPrev,
    WriteVaspFromIOSetFromInterpolatedPOSCAR,
    UpdateScanRelaxBandgap,
    ModifyIncar
)
from atomate.vasp.firetasks.neb_tasks import WriteNEBFromImages, WriteNEBFromEndpoints
from atomate.vasp.firetasks.jcustom import RmSelectiveDynPoscar, SelectiveDynmaicPoscar, \
    JWriteScanVaspStaticFromPrev
from atomate.vasp.config import VASP_CMD, DB_FILE


class JScanOptimizeFW(Firework):
    def __init__(
            self,
            structure,
            name="JSCAN structure optimization",
            vasp_input_set=None,
            vasp_cmd=VASP_CMD,
            override_default_vasp_params=None,
            ediffg=None,
            db_file=DB_FILE,
            force_gamma=True,
            job_type="double_relaxation_run",
            max_force_threshold=RELAX_MAX_FORCE,
            auto_npar=">>auto_npar<<",
            half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX,
            parents=None,
            **kwargs
    ):
        """
        Optimize the given structure.

        Args:
            structure (Structure): Input structure.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
            override_default_vasp_params (dict): If this is not None, these params are passed to
                the default vasp_input_set, i.e., MPRelaxSet. This allows one to easily override
                some settings, e.g., user_incar_settings, etc.
            vasp_cmd (str): Command to run vasp.
            ediffg (float): Shortcut to set ediffg in certain jobs
            db_file (str): Path to file specifying db credentials to place output parsing.
            force_gamma (bool): Force gamma centered kpoint generation
            job_type (str): custodian job type (default "double_relaxation_run")
            max_force_threshold (float): max force on a site allowed at end; otherwise, reject job
            auto_npar (bool or str): whether to set auto_npar. defaults to env_chk: ">>auto_npar<<"
            half_kpts_first_relax (bool): whether to use half the kpoints for the first relaxation
            parents ([Firework]): Parents of this particular Firework.
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        override_default_vasp_params = override_default_vasp_params or {}
        vasp_input_set = vasp_input_set or MVLScanRelaxSet(
            structure, force_gamma=force_gamma, **override_default_vasp_params
        )

        if vasp_input_set.incar["ISIF"] in (0, 1, 2, 7) and job_type == "double_relaxation":
            warnings.warn(
                "A double relaxation run might not be appropriate with ISIF {}".format(
                    vasp_input_set.incar["ISIF"]))

        t = []
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))
        t.append(
            RunVaspCustodian(
                vasp_cmd=vasp_cmd,
                job_type=job_type,
                max_force_threshold=max_force_threshold,
                ediffg=ediffg,
                auto_npar=auto_npar,
                half_kpts_first_relax=half_kpts_first_relax,
            )
        )
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, additional_fields={"task_label": name}))
        super(JScanOptimizeFW, self).__init__(
            t,
            parents=parents,
            name="{}-{}".format(structure.composition.reduced_formula, name),
            **kwargs
        )


class JScanStaticFW(Firework):
    def __init__(
            self,
            structure=None,
            name="JScan static",
            vasp_input_set=None,
            vasp_input_set_params=None,
            vasp_cmd=VASP_CMD,
            prev_calc_loc=True,
            prev_calc_dir=None,
            db_file=DB_FILE,
            vasptodb_kwargs=None,
            parents=None,
            **kwargs
    ):
        """
        Standard static calculation Firework - either from a previous location or from a structure.

        Args:
            structure (Structure): Input structure. Note that for prev_calc_loc jobs, the structure
                is only used to set the name of the FW and any structure with the same composition
                can be used.
            name (str): Name for the Firework.
            vasp_input_set (VaspInputSet): input set to use (for jobs w/no parents)
                Defaults to MPStaticSet() if None.
            vasp_input_set_params (dict): Dict of vasp_input_set kwargs.
            vasp_cmd (str): Command to run vasp.
            prev_calc_loc (bool or str): If true (default), copies outputs from previous calc. If
                a str value, retrieves a previous calculation output by name. If False/None, will create
                new static calculation using the provided structure.
            prev_calc_dir (str): Path to a previous calculation to copy from
            db_file (str): Path to file specifying db credentials.
            parents (Firework): Parents of this particular Firework. FW or list of FWS.
            vasptodb_kwargs (dict): kwargs to pass to VaspToDb
            \*\*kwargs: Other kwargs that are passed to Firework.__init__.
        """
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(
            structure.composition.reduced_formula if structure else "unknown", name
        )

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(JWriteScanVaspStaticFromPrev(other_params=vasp_input_set_params))
        elif parents:
            if prev_calc_loc:
                t.append(
                    CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True)
                )
            t.append(JWriteScanVaspStaticFromPrev(other_params=vasp_input_set_params))
        elif structure:
            vasp_input_set = vasp_input_set or MPScanStaticSet(
                structure, **vasp_input_set_params
            )
            t.append(
                WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set)
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(JScanStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class HSEStaticFW(Firework):
    def __init__(self, structure=None, name="HSE_scf", vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, prev_calc_dir=None, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, cp_chargcar=True, force_gamma=True, **kwargs):
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(
                CopyVaspOutputs(calc_dir=prev_calc_dir, additional_files=["CHGCAR"] if cp_chargcar else [])
            )
        elif parents:
            t.append(CopyVaspOutputs(calc_loc=True, additional_files=["CHGCAR"] if cp_chargcar else []))
        else:
            raise ValueError("Must specify structure or previous calculation")
        t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
        t.append(RmSelectiveDynPoscar())

        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))

        if vasp_input_set_params.get("user_incar_settings", {}):
            t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))
        if vasp_input_set_params.get("user_kpoints_settings", {}):
            t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        else:
            t.append(WriteVaspFromPMGObjects(
                kpoints=MPHSERelaxSet(structure=structure, force_gamma=force_gamma).kpoints.as_dict()))
        
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, bandstructure_mode="uniform", parse_eigenvalues=True, parse_dos=True,
                          **vasptodb_kwargs))
        super(HSEStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class HSERelaxFW(Firework):
    def __init__(self, structure=None, name="HSE_relax", vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, wall_time=None, force_gamma=True, **kwargs):
        t = []
        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if parents:
            t.append(CopyVaspOutputs(calc_loc=True)) #, additional_files=["CHGCAR"]))
        else:
            raise ValueError("Must specify the parent")
        t.append(WriteVaspStaticFromPrev())
        hse_relax_vis_incar = MPHSERelaxSet(structure=structure).incar
        t.append(ModifyIncar(incar_update=hse_relax_vis_incar))

        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))
            
        if vasp_input_set_params.get("user_incar_settings", {}):
            t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))

        if vasp_input_set_params.get("user_kpoints_settings", {}):
            t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        else:
            t.append(WriteVaspFromPMGObjects(kpoints=MPHSERelaxSet(structure=structure, 
                                                                   force_gamma=force_gamma).kpoints.as_dict()))
        
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", max_errors=5, wall_time=wall_time))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(HSERelaxFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class HSEcDFTFW(Firework):
    def __init__(self, prev_calc_dir, structure=None, read_structure_from=None, name="HSE_cDFT",
                 vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, selective_dynamics=None, **kwargs):

        t = []
        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if read_structure_from:
            t.append(CopyVaspOutputs(additional_files=["WAVECAR"], calc_dir=read_structure_from))
        elif parents:
            t.append(CopyVaspOutputs(additional_files=["WAVECAR"], calc_dir=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
        else:
            t.append(CopyVaspOutputs(additional_files=["WAVECAR"], calc_dir=prev_calc_dir))
            t.append(WriteVaspHSEBSFromPrev(prev_calc_dir=prev_calc_dir,
                                            mode="uniform", reciprocal_density=None, kpoints_line_density=None))
        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))
        t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))
        if selective_dynamics:
            t.append(SelectiveDynmaicPoscar(selective_dynamics=selective_dynamics, nsites=len(structure.sites)))
        t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, bandstructure_mode="uniform", parse_dos=True,
                          parse_eigenvalues=True, **vasptodb_kwargs))
        super(HSEcDFTFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class PBEcDFTRelaxFW(Firework):
    def __init__(self, prev_calc_dir, vis="MPRelaxSet", structure=None, read_structure_from=None, name="cDFT_PBE_relax",
                 vasp_input_set_params=None, vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, wall_time=None, **kwargs):
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        if read_structure_from:
            t.append(CopyVaspOutputs(additional_files=["WAVECAR"], calc_dir=read_structure_from, contcar_to_poscar=True))
        else:
            t.append(CopyVaspOutputs(additional_files=["WAVECAR"], calc_dir=prev_calc_dir))
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vis))
        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))
        t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))
        t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", max_errors=5, wall_time=wall_time))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(PBEcDFTRelaxFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class PBEcDFTStaticFW(Firework):
    def __init__(self, structure, name="cDFT_PBE_scf",
                 vasp_input_set_params=None, vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, wall_time=None, **kwargs):
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)
        if parents:
            t.append(CopyVaspOutputs(calc_loc=True, contcar_to_poscar=True))
        else:
            t.append(CopyVaspOutputs(additional_files=["CHGCAR"], calc_loc=True))
        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))
        t.append(WriteVaspStaticFromPrev())
        t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))
        t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", max_errors=5, wall_time=wall_time))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, bandstructure_mode="uniform",
                          parse_dos=True, parse_eigenvalues=True, **vasptodb_kwargs))
        super(PBEcDFTStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)
