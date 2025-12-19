import dendropy

posterior_path = "long_v3_44.trees"

trees_speech = dendropy.TreeList.get(
    path=posterior_path,
    schema="nexus",
    preserve_underscores=True
)

posterior_path = "IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined.trees"

trees_cognates = dendropy.TreeList.get(
    path=posterior_path,
    schema="nexus",
    preserve_underscores=True
)

# prune trees to only those languages in trees_speech
# prune Hittite and Luvian from trees_cognates

def treelist_taxa_labels(tree_list):
    """
    Get the set of taxon labels present in (the first) tree_list.
    Assumes posterior: all trees have the same taxa.
    """
    if len(tree_list) == 0:
        return set()
    return {leaf.taxon.label for leaf in tree_list[0].leaf_node_iter()}


def prune_treelist_to_labels(tree_list, keep_labels):
    """
    Return a *new* TreeList where each tree is pruned
    to only taxa whose labels are in `keep_labels`.
    """
    pruned = dendropy.TreeList()
    for tr in tree_list:
        tr2 = tr.clone(depth=2)
        tr2.retain_taxa_with_labels(keep_labels)
        pruned.append(tr2)
    return pruned

def remove_taxa_from_treelist(tree_list, remove_labels):
    """
    Return a *new* TreeList where each tree has the specified taxa removed.
    Uses prune_taxa_with_labels (removes listed) vs retain_taxa_with_labels (keeps listed).
    """
    pruned = dendropy.TreeList()
    for tr in tree_list:
        tr2 = tr.clone(depth=2)
        tr2.prune_taxa_with_labels(remove_labels)
        tr2.suppress_unifurcations()  # Clean up degree-2 nodes
        pruned.append(tr2)
    return pruned


# 1) Get label sets
speech_labels   = treelist_taxa_labels(trees_speech)
cognate_labels  = treelist_taxa_labels(trees_cognates)

overlap         = speech_labels & cognate_labels
missing_in_cogn = speech_labels - cognate_labels
extra_in_cogn   = cognate_labels - speech_labels

print(f"Speech taxa:   {len(speech_labels)}")
print(f"Cognate taxa:  {len(cognate_labels)}")
print(f"Overlap:       {len(overlap)}")
print(f"Missing in cognates (in speech but not cognates): {sorted(missing_in_cogn)}")
print(f"Extra in cognates (not in speech):                {sorted(extra_in_cogn)}")

# 2) Prune cognate trees down to overlap
pruned_cognates = prune_treelist_to_labels(trees_cognates, overlap)
pruned_cognates.write(
    path="IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined_prunedtospeech.trees",
    schema="nexus"
)

print("Example tree sizes before/after pruning:")
print("  original cognate taxa:", len(cognate_labels))
print("  pruned cognate taxa:  ",
      len({leaf.taxon.label for leaf in pruned_cognates[0].leaf_node_iter()}))
# Quick sanity check
orig_total = sum(nd.edge.length for nd in trees_cognates[0].postorder_node_iter() 
                 if nd.edge.length is not None)
pruned_total = sum(nd.edge.length for nd in pruned_cognates[0].postorder_node_iter() 
                   if nd.edge.length is not None)
print(f"Total branch length - original: {orig_total:.2f}, pruned: {pruned_total:.2f}")

pruned_cognates.write(
    path="IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_combined_prunedtospeech.trees",
    schema="nexus" 
)