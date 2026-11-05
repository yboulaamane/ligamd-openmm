# Cutting the release the paper cites

Everything below runs on your machine; none of it can be done from the cluster
session, which has no GitHub credentials.

## 1. Push the outstanding commit

`09b1f8b` ("Use the Ewald real-space term for the isolated ligand
electrostatics") is committed locally but not pushed.

```bash
git push origin fix/ligand-group-electrostatics
```

Then merge it to the default branch, since the paper's Figure 3 describes the
behaviour of that commit and the released code must match.

## 2. Tag

```bash
git tag -a v0.1.0 -m "First release: ligand-selective boost, Ewald real-space ligand electrostatics"
git push origin v0.1.0
```

## 3. Connect Zenodo

1. Sign in at zenodo.org with GitHub.
2. Under **GitHub**, flip the switch on `yboulaamane/ligamd-openmm`.
3. Re-publish the release on GitHub (Releases -> Draft a new release -> pick
   `v0.1.0`). Zenodo only mints a DOI for releases created *after* the switch is
   on, so if the tag was pushed first, cut the GitHub release afterwards.
4. Zenodo issues both a version DOI and a concept DOI. **Cite the concept DOI**
   in the paper: it always resolves to the newest version.

## 4. Put the DOI in three places

- `CITATION.cff` -> add `doi:` and `identifiers:`
- `README.md` -> DOI badge
- the manuscript, Data and code availability

## Note on the licence

Upstream gamd-openmm is MIT, so redistributing this fork under MIT is fine.
Keep the upstream copyright line in `LICENSE` alongside yours; removing it would
breach the MIT terms.
