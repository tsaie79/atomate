import warnings

from fireworks import Firework

from pymatgen import Structure
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPHSERelaxSet,
    MPHSEBSSet,
    MPScanStaticSet,
    MPScanRelaxSet,
    MVLScanRelaxSet,
    MVLGWSet,
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
    JWriteScanVaspStaticFromPrev, JWriteMVLGWFromPrev
from atomate.vasp.config import VASP_CMD, DB_FILE

class JOptimizeFW(Firework):
    def __init__(
            self,
            structure,
            name="structure optimization",
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
            vasptodb_kwargs=None,
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
        vasp_input_set = vasp_input_set or MPRelaxSet(
            structure, force_gamma=force_gamma, **override_default_vasp_params
        )

        if vasp_input_set.incar["ISIF"] in (0, 1, 2, 7) and job_type == "double_relaxation":
            warnings.warn(
                "A double relaxation run might not be appropriate with ISIF {}".format(
                    vasp_input_set.incar["ISIF"]))

        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

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
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(JOptimizeFW, self).__init__(
            t,
            parents=parents,
            name="{}-{}".format(structure.composition.reduced_formula, name),
            **kwargs
        )

class JSelectiveOptFW(Firework):
    """
    Copy from OptimizeFW completely except adding firetask SelectiveDynmaicPoscar
    """
    def __init__(
            self,
            structure,
            name="structure optimization",
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
            vasptodb_kwargs=None,
            selective_dynamics=None,
            prev_calc_loc=True,
            **kwargs
    ):

        override_default_vasp_params = override_default_vasp_params or {}
        vasp_input_set = vasp_input_set or MPRelaxSet(
            structure, force_gamma=force_gamma, **override_default_vasp_params
        )

        if vasp_input_set.incar["ISIF"] in (0, 1, 2, 7) and job_type == "double_relaxation":
            warnings.warn(
                "A double relaxation run might not be appropriate with ISIF {}".format(
                    vasp_input_set.incar["ISIF"]))
        t = []
        if parents:
            if prev_calc_loc:
                t.append(
                    CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True)
                )
            mprelax_incar = MPRelaxSet(structure, force_gamma=force_gamma, **override_default_vasp_params).incar.as_dict()
            mprelax_incar.pop("@module")
            mprelax_incar.pop("@class")
            t.append(WriteVaspStaticFromPrev(other_params={"user_incar_settings": mprelax_incar}))

        elif structure:
            t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))

        if selective_dynamics:
            t.append(SelectiveDynmaicPoscar(selective_dynamics=selective_dynamics, nsites=len(structure.sites)))
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
        t.append(VaspToDb(db_file=db_file, additional_fields={"task_label": name}, **vasptodb_kwargs))
        super(JSelectiveOptFW, self).__init__(
            t,
            parents=parents,
            name="{}-{}".format(structure.composition.reduced_formula, name),
            **kwargs
        )



class JMVLGWFW(Firework):
    def __init__(
            self,
            structure,
            mode,
            name="J MVLGW",

            prev_incar=None,
            nbands=None,
            reciprocal_density=100,
            nbands_factor=5,
            ncores=None,

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

        additional_file = []

        if mode == "DIAG":
            additional_file.append("WAVECAR")
        elif mode == "GW":
            additional_file.append("WAVECAR")
            additional_file.append("WAVEDER")
        elif mode == "BSE":
            additional_file.append("WAVECAR")
            additional_file.append("WAVEDER")
            additional_file.append("WFULL")


        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True, additional_files=additional_file))
            t.append(JWriteMVLGWFromPrev(nbands=nbands, reciprocal_density=reciprocal_density,
                                         nbands_factor=nbands_factor, ncores=ncores, prev_incar=prev_incar,
                                         mode=mode, other_params=vasp_input_set_params))
        elif parents:
            if prev_calc_loc:
                t.append(
                    CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True, additional_files=additional_file)
                )
            t.append(JWriteMVLGWFromPrev(nbands=nbands, reciprocal_density=reciprocal_density,
                                         nbands_factor=nbands_factor, ncores=ncores, prev_incar=prev_incar,
                                         mode=mode, other_params=vasp_input_set_params))
        elif structure:
            vasp_input_set = vasp_input_set or MVLGWSet(
                structure, **vasp_input_set_params
            )
            t.append(
                WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set)
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<", handler_group="no_handler"))
        t.append(PassCalcLocs(name=name))
        # t.append(VaspToDb(db_file=db_file, defuse_unsuccessful=True, **vasptodb_kwargs))
        super(JMVLGWFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


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
            vasptodb_kwargs=None,
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

        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        t = []
        t.append(WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set))

        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))

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
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(JScanOptimizeFW, self).__init__(
            t,
            parents=parents,
            name=fw_name,
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
            force_gamma=True,
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
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(JWriteScanVaspStaticFromPrev(other_params=vasp_input_set_params))
        elif structure:
            vasp_input_set = vasp_input_set or "MPScanStaticSet"
            t.append(
                WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set)
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RmSelectiveDynPoscar())

        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))

        t.append(ModifyIncar(incar_update={"EDIFF": 1E-5}))

        if vasp_input_set_params.get("user_incar_settings", {}):
            t.append(ModifyIncar(incar_update=vasp_input_set_params.get("user_incar_settings", {})))

        if vasp_input_set_params.get("user_kpoints_settings", {}):
            t.append(WriteVaspFromPMGObjects(kpoints=vasp_input_set_params.get("user_kpoints_settings", {})))
        else:
            t.append(WriteVaspFromPMGObjects(
                kpoints=MPRelaxSet(structure=structure, force_gamma=force_gamma).kpoints.as_dict()))

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<"))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(JScanStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class JHSEStaticFW(Firework):
    def __init__(self, structure=None, name="HSE_scf", vasp_input_set=None, vasp_input_set_params=None,
                 vasp_cmd=VASP_CMD, prev_calc_loc=True, prev_calc_dir=None, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, force_gamma=True, **kwargs):
        t = []

        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
        elif structure:
            vasp_input_set = vasp_input_set or "MPHSERelaxSet"
            incar_hse_bs = MPHSEBSSet(structure).incar.as_dict()
            for x in ['@module', '@class', "MAGMOM"]:
                incar_hse_bs.pop(x)
            t.append(
                WriteVaspFromIOSet(
                    structure=structure,
                    vasp_input_set=vasp_input_set,
                    vasp_input_params={"user_incar_settings": incar_hse_bs}
                )
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

        t.append(RmSelectiveDynPoscar())

        magmom = MPRelaxSet(structure).incar.get("MAGMOM", None)
        if magmom:
            t.append(ModifyIncar(incar_update={"MAGMOM": magmom}))

        t.append(ModifyIncar(incar_update={"EDIFF": 1E-5}))

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
        super(JHSEStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class JHSERelaxFW(Firework):
    def __init__(
            self,
            structure=None,
            name="HSE_relax",
            vasp_input_set_params=None,
            vasp_input_set=None,
            vasp_cmd=VASP_CMD,
            prev_calc_loc=True,
            prev_calc_dir=None,
            db_file=DB_FILE,
            vasptodb_kwargs=None,
            parents=None,
            force_gamma=True,
            job_type="normal",
            max_force_threshold=False,
            ediffg=None,
            auto_npar=">>auto_npar<<",
            half_kpts_first_relax=HALF_KPOINTS_FIRST_RELAX,
            **kwargs
    ):

        t = []
        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        hse_relax_vis_incar = MPHSERelaxSet(structure=structure).incar

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
            t.append(ModifyIncar(incar_update=hse_relax_vis_incar))
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
            t.append(ModifyIncar(incar_update=hse_relax_vis_incar))
        elif structure:
            vasp_input_set = vasp_input_set or "MPHSERelaxSet"
            t.append(
                WriteVaspFromIOSet(structure=structure, vasp_input_set=vasp_input_set)
            )
        else:
            raise ValueError("Must specify structure or previous calculation")

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
        t.append(VaspToDb(db_file=db_file, **vasptodb_kwargs))
        super(JHSERelaxFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class JHSEcDFTFW(Firework):

    def __init__(self, up_occupation, down_occupation, nbands, prev_calc_dir=None, structure=None, name="HSE_cDFT",
                 vasp_input_set_params=None, job_type="normal", max_force_threshold=None,
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, prev_calc_loc=True, selective_dynamics=None, force_gamma=True, **kwargs):

        t = []
        vasp_input_set_params = vasp_input_set_params or {}
        vasptodb_kwargs = vasptodb_kwargs or {}
        if "additional_fields" not in vasptodb_kwargs:
            vasptodb_kwargs["additional_fields"] = {}
        vasptodb_kwargs["additional_fields"]["task_label"] = name

        fw_name = "{}-{}".format(structure.composition.reduced_formula if structure else "unknown", name)

        if prev_calc_dir:
            t.append(CopyVaspOutputs(calc_dir=prev_calc_dir, additional_files=["WAVECAR"], contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
            t.append(RmSelectiveDynPoscar())
        elif parents:
            if prev_calc_loc:
                t.append(CopyVaspOutputs(calc_loc=prev_calc_loc, additional_files=["WAVECAR"], contcar_to_poscar=True))
            t.append(WriteVaspHSEBSFromPrev(mode="uniform", reciprocal_density=None, kpoints_line_density=None))
            t.append(RmSelectiveDynPoscar())
        else:
            raise ValueError("Must specify previous calculation or parent")

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

        if selective_dynamics:
            t.append(SelectiveDynmaicPoscar(selective_dynamics=selective_dynamics, nsites=len(structure.sites)))

        t.append(ModifyIncar(incar_update={
            "ISMEAR": -2,
            "FERWE": up_occupation,
            "FERDO": down_occupation,
            "NBANDS": nbands,
            "LDIAG": False,
            "LSUBROT": True,
            "ALGO": "All"
        }))

        t.append(RunVaspCustodian(vasp_cmd=vasp_cmd, auto_npar=">>auto_npar<<",
                                  job_type=job_type, max_force_threshold=max_force_threshold))
        t.append(PassCalcLocs(name=name))
        t.append(VaspToDb(db_file=db_file, bandstructure_mode="uniform", **vasptodb_kwargs))
        super(JHSEcDFTFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class JPBEcDFTRelaxFW(Firework):
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
        super(JPBEcDFTRelaxFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)


class JPBEcDFTStaticFW(Firework):
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
        super(JPBEcDFTStaticFW, self).__init__(t, parents=parents, name=fw_name, **kwargs)
