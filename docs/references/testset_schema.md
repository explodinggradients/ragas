::: ragas.testset.synthesizers.testset_schema
    options:
        members_order: "source"

::: ragas.testset.synthesizers.base
    options:
        members:
            - QueryLength
            - QueryStyle

::: ragas.testset.synthesizers.base.Scenario

::: ragas.testset.synthesizers.base
    options:
        members:
            - BaseScenario

::: ragas.testset.synthesizers.specific_query.SpecificQueryScenario
    options:
        show_root_heading: True
        show_root_full_path: False

::: ragas.testset.synthesizers.abstract_query.AbstractQueryScenario
    options:
        show_root_heading: True
        show_root_full_path: False
