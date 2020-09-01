from fireworks import FiretaskBase, explicit_serialize
from pymatgen.io.vasp.inputs import Structure, Poscar

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