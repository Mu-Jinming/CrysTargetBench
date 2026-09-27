"""Versioned serial recipe priority shared by planning and local execution."""
VERSION = 'ctb.serial_recipe_schedule.v1'


def measurement_schedule(task):
    def key(alias):
        prop=task['measurements'][alias]['property_id']
        rank=(0 if prop=='band_gap_eV' else 10 if prop.startswith('phonon_') else
              20 if prop.startswith('dielectric_') else 30 if prop=='space_group_number' else 40)
        return rank,prop,alias
    return {'version':VERSION,'measurement_order':sorted(task['measurements'],key=key),
            'priority_groups':['band_gap','phonon','dielectric','space_group','other_including_bulk'],
            'tie_break':'property_group_then_property_id_then_alias_unicode',
            'reference_dependency':'validate_then_relax_or_check_then_freeze_before_measurements',
            's2_local_gates':{'reference_unqualified':'unknown_all',
                'measurement_failed_or_unknown':'continue_independent_measurements',
                'measurement_accepted_fail':'continue_independent_measurements',
                'eos_point_unqualified':'skip_remaining_points_no_fit'},
            'dft_response_gates':'explicit_DAG_conditions_preserved'}
