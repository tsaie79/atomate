from atomate.common.firetasks.glue_tasks import DeleteFiles
from atomate.utils.utils import get_meta_from_structure, get_fws_and_tasks
from atomate.vasp.config import (
    ADD_NAMEFILE,
    SCRATCH_DIR,
    ADD_MODIFY_INCAR,
    GAMMA_VASP_CMD,
)
from atomate.vasp.firetasks.jcustom import JFileTransferTask, JWriteInputsFromDB
from atomate.vasp.firetasks.glue_tasks import CheckStability, CheckBandgap
from atomate.vasp.firetasks.lobster_tasks import RunLobsterFake
from atomate.vasp.firetasks.neb_tasks import RunNEBVaspFake
from atomate.vasp.firetasks.parse_outputs import JsonToDb
from atomate.vasp.firetasks.run_calc import (
    RunVaspCustodian,
    RunVaspFake,
    RunVaspDirect,
    RunNoVasp,
)
from atomate.vasp.firetasks.write_inputs import ModifyIncar, ModifyPotcar, ModifyKpoints
from fireworks import Workflow, FileWriteTask
from fireworks.core.firework import Tracker
from fireworks.utilities.fw_utilities import get_slug
from pymatgen import Structure

import os

__author__ = "Jeng-Yuan Tsai"
__email__ = "tsaie79@gmail.com"


def scp_files(
        original_wf,
        root_path,
        proj_name,
        calc_name,
        fw_name_constraint=None,
        task_name_constraint="RunVasp",
):
    """
    SCP ALL files to local computer

    Args:
        original_wf (Workflow)
        root_path (str): "/home/jengyuantsai/test_scp_fw/"
        proj_name (str): "/home/jengyuantsai/test_scp_fw/defect_db/"
        calc_name (str): "/home/jengyuantsai/test_scp_fw/defect_db/binary_vac_AB/"
        fw_name_constraint (str): pattern for fireworks to clean up files after
        task_name_constraint (str): pattern for firetask to clean up files

    Returns:
       Workflow
    """
    idx_list = get_fws_and_tasks(
        original_wf,
        fw_name_constraint=fw_name_constraint,
        task_name_constraint=task_name_constraint,
    )
    for idx_fw, idx_t in idx_list:
        original_wf.fws[idx_fw].tasks.insert(idx_t + 1, JFileTransferTask(
            mode="rtransfer",
            files=["all"],
            dest=os.path.join(root_path, proj_name, calc_name),
            server="localhost",
            user="jengyuantsai",
            key_filename=">>ssh_key<<"
        ))

    return original_wf

def write_inputs_from_db(original_wf, db_file, task_id, modify_incar, write_chgcar=True, fw_name_constraint=None):

    idx_list = get_fws_and_tasks(
        original_wf,
        fw_name_constraint=fw_name_constraint,
        task_name_constraint="RunVasp",
    )
    for idx_fw, idx_t in idx_list:
        original_wf.fws[idx_fw].tasks.insert(idx_t - 1, JWriteInputsFromDB(db_file=db_file, task_id=task_id,
                                                                           write_chgcar=write_chgcar,
                                                                           modify_incar=modify_incar))
    return original_wf