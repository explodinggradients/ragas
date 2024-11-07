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

::: ragas.testset.synthesizers.single_hop.specific.SingleHopSpecificQuerySynthesizer
    options:
        show_root_heading: True
        show_root_full_path: False

::: ragas.testset.synthesizers.multi_hop.specific.MultiHopSpecificQuerySynthesizer
    options:
        show_root_heading: True
        show_root_full_path: False
