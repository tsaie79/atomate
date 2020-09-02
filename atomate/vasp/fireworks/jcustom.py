from fireworks import Firework

from pymatgen import Structure
from pymatgen.io.vasp.sets import (
    MPRelaxSet,
    MPHSERelaxSet,
    MPScanRelaxSet,
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
from atomate.vasp.firetasks.jcustom import RmSelectiveDynPoscar, SelectiveDynmaicPoscar
from atomate.vasp.config import VASP_CMD, DB_FILE

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

        if MPRelaxSet(structure).incar.get("MAGMOM", None):
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
    def __init__(self, structure=None, name="HSE_relax", vasp_input_set_params={},
                 vasp_cmd=VASP_CMD, db_file=DB_FILE, vasptodb_kwargs=None,
                 parents=None, wall_time=None, force_gamma=True, **kwargs):
        t = []
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

        if MPRelaxSet(structure).incar.get("MAGMOM", None):
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