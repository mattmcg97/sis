#!/usr/bin/env Rscript
#
# Install what the glmer scripts need: lme4 (which brings nloptr, the
# optimiser behind glmerControl(optimizer = "nloptwrap")). CRAN has Windows
# and macOS binaries for both, so no compiler is needed.
#
#   Rscript glmer/install_packages.R

pkgs <- c("lme4", "nloptr")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}
for (p in pkgs) cat(sprintf("%-8s %s\n", p, as.character(packageVersion(p))))
