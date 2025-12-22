#!/usr/bin/env Rscript

# run_linear_single.R
#
# Runs a SINGLE tree × interaction_level combination
#
# Usage:
#   Rscript speech_phylo/final_scripts/run_linear_single.R <tree_name> <interaction_level>
#
# Example:
#   Rscript speech_phylo/final_scripts/run_linear_single.R long_v3_CCD0.0.25 main_only

args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 2) {
    stop("Usage: Rscript run_linear_single.R <tree_name> <interaction_level>")
}

tree_name <- args[1]
int_level <- args[2]
coord_method <- "standard"

cat(sprintf("═══════════════════════════════════════════════════════════════════\n"))
cat(sprintf("Running: tree=%s, level=%s\n", tree_name, int_level))
cat(sprintf("═══════════════════════════════════════════════════════════════════\n\n"))

# ─── Setup ─────────────────────────────────────────────────────────
suppressPackageStartupMessages({
    library(ape)
    library(brms)
    library(dplyr)
    library(tibble)
})

out_dir <- "speech_phylo/final_phyloregression_results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

source("speech_phylo/beast.R")

# ─── Helper functions ──────────────────────────────────────────────
get_formula <- function(level) {
    base_terms <- "longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm"
    
    second_order <- paste0(
        "longitude_norm:latitude_norm + ",
        "longitude_norm:log_n_speakers_norm + ",
        "longitude_norm:n_segments_norm + ",
        "longitude_norm:delta_norm + ",
        "latitude_norm:log_n_speakers_norm + ",
        "latitude_norm:n_segments_norm + ",
        "latitude_norm:delta_norm + ",
        "log_n_speakers_norm:n_segments_norm + ",
        "log_n_speakers_norm:delta_norm + ",
        "n_segments_norm:delta_norm")
    
    third_order <- paste0(
        "longitude_norm:latitude_norm:log_n_speakers_norm + ",
        "longitude_norm:latitude_norm:n_segments_norm + ",
        "longitude_norm:latitude_norm:delta_norm + ",
        "longitude_norm:log_n_speakers_norm:n_segments_norm + ",
        "longitude_norm:log_n_speakers_norm:delta_norm + ",
        "longitude_norm:n_segments_norm:delta_norm + ",
        "latitude_norm:log_n_speakers_norm:n_segments_norm + ",
        "latitude_norm:log_n_speakers_norm:delta_norm + ",
        "latitude_norm:n_segments_norm:delta_norm + ",
        "log_n_speakers_norm:n_segments_norm:delta_norm")
    
    fourth_order <- paste0(
        "longitude_norm:latitude_norm:log_n_speakers_norm:n_segments_norm + ",
        "longitude_norm:latitude_norm:log_n_speakers_norm:delta_norm + ",
        "longitude_norm:latitude_norm:n_segments_norm:delta_norm + ",
        "longitude_norm:log_n_speakers_norm:n_segments_norm:delta_norm + ",
        "latitude_norm:log_n_speakers_norm:n_segments_norm:delta_norm")
    
    fifth_order <- "longitude_norm:latitude_norm:log_n_speakers_norm:n_segments_norm:delta_norm"
    
    if (level == "main_only") {
        rhs <- base_terms
    } else if (level == "secondorder") {
        rhs <- paste0(base_terms, " + ", second_order)
    } else if (level == "thirdorder") {
        rhs <- paste0(base_terms, " + ", second_order, " + ", third_order)
    } else if (level == "fourthorder") {
        rhs <- paste0(base_terms, " + ", second_order, " + ", third_order, " + ", fourth_order)
    } else if (level == "fifthorder") {
        rhs <- paste0(base_terms, " + ", second_order, " + ", third_order, " + ", fourth_order, " + ", fifth_order)
    }
    
    formula_str <- paste0("log_rate_median ~ ", rhs, " + (1 | gr(language_factor, cov = V))")
    return(bf(as.formula(formula_str), decomp = "QR"))
}

get_design_formula <- function(level) {
    if (level == "main_only") {
        return(~ longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm)
    } else if (level == "secondorder") {
        return(~ (longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm)^2)
    } else if (level == "thirdorder") {
        return(~ (longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm)^3)
    } else if (level == "fourthorder") {
        return(~ (longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm)^4)
    } else if (level == "fifthorder") {
        return(~ (longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm)^5)
    }
}

apply_coord_scaling <- function(vec, method, shift = 10) {
    if (method == "standard") {
        transformed <- vec
    } else if (method == "sqrt_plus_10") {
        transformed <- sqrt(vec + shift)
    } else if (method == "log_plus_10") {
        transformed <- log(vec + shift)
    }
    scaled <- scale(transformed)
    return(list(
        values = as.numeric(scaled),
        center = as.numeric(attr(scaled, "scaled:center")),
        scale = as.numeric(attr(scaled, "scaled:scale"))
    ))
}

summarize_posterior <- function(x) {
    list(
        mean  = mean(x, na.rm = TRUE),
        sd    = sd(x, na.rm = TRUE),
        q2.5  = unname(quantile(x, 0.025, na.rm = TRUE)),
        q50   = unname(quantile(x, 0.50, na.rm = TRUE)),
        q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
    )
}

# ─── Load data ─────────────────────────────────────────────────────
tree_path <- paste0("speech_phylo/", tree_name, ".nex")
tr <- read.annot.beast(tree_path)

reg_data_path <- file.path(out_dir, paste0("metadata_", tree_name, "_with_inventory_delta.csv"))
df <- read.csv(reg_data_path, stringsAsFactors = FALSE)
rownames(df) <- df$language

tr <- drop.tip(tr, setdiff(tr$tip.label, rownames(df)))
V_raw <- vcv(unroot(tr))
V_raw <- V_raw[rownames(df), rownames(df)]

log_n_scaled <- scale(log(df$n_speakers))
df$log_n_speakers_norm <- log_n_scaled[, 1]

lon_stats <- apply_coord_scaling(df$longitude, coord_method)
lat_stats <- apply_coord_scaling(df$latitude, coord_method)
df$longitude_norm <- lon_stats$values
df$latitude_norm <- lat_stats$values

df$n_segments_norm <- scale(df$n_segments)[, 1]
df$delta_norm <- scale(df$delta)[, 1]

model_df <- df
model_df$log_rate_median <- log(model_df$rate_median)
language_levels <- rownames(model_df)
model_df$language_factor <- factor(language_levels, levels = language_levels)

V <- V_raw[language_levels, language_levels]

# ─── Fit model ─────────────────────────────────────────────────────
formula_obj <- get_formula(int_level)

# priors <- c(
#     set_prior("normal(0, 0.3)", class = "b"),
#     set_prior("student_t(3, -6, 1)", class = "Intercept"),
#     set_prior("exponential(1)", class = "sigma"),
#     set_prior("exponential(1)", class = "sd", group = "language_factor")
# )

brm_args <- list(
    formula = formula_obj,
    data = model_df,
    family = gaussian(),
    # prior = priors,
    refresh = 500,
    chains = 4,
    iter = 30000,
    warmup = 10000,
    seed = 20231103,
    cores = 4,
    control = list(adapt_delta = 0.9999, max_treedepth = 15),
    data2 = list(V = V),
    sample_prior = "no",
    thin = 10
)

model <- do.call(brm, brm_args)

# ─── Diagnostics ───────────────────────────────────────────────────
cat("\n======================================================================\n")
cat("                    MCMC DIAGNOSTICS\n")
cat("======================================================================\n")

# Divergent transitions
np <- nuts_params(model)
n_div <- sum(np[np$Parameter == "divergent__", "Value"])
n_samples <- (brm_args$iter - brm_args$warmup) * brm_args$chains
cat(sprintf("Divergent transitions: %d / %d (%.2f%%)\n", n_div, n_samples, 100 * n_div / n_samples))
if (n_div > 0) {
    cat("  [!] Divergences detected - consider increasing adapt_delta\n")
}

# Tree depth
max_td <- sum(np[np$Parameter == "treedepth__", "Value"] >= brm_args$control$max_treedepth)
cat(sprintf("Max treedepth reached: %d times (max=%d)\n", max_td, brm_args$control$max_treedepth))

# BFMI (Bayesian Fraction of Missing Information)  
# Low BFMI (<0.2) indicates poor exploration
energy <- np[np$Parameter == "energy__", "Value"]
if (length(energy) > 0) {
    chain_ids <- np[np$Parameter == "energy__", "Chain"]
    chains <- unique(chain_ids)
    bfmi_vals <- sapply(chains, function(ch) {
        e <- energy[chain_ids == ch]
        var(diff(e)) / var(e)
    })
    cat(sprintf("BFMI per chain: %s\n", paste(sprintf("%.3f", bfmi_vals), collapse=", ")))
    if (any(bfmi_vals < 0.2)) {
        cat("  [!] Low BFMI (<0.2) detected - may indicate poor posterior exploration\n")
    }
}

# Rhat and ESS for ALL parameters (fixed + random + sigma)
summ_fixed <- summary(model)$fixed
summ_spec <- summary(model)$spec_pars
summ_random <- tryCatch(summary(model)$random$language_factor, error = function(e) NULL)

all_rhat <- c(summ_fixed$Rhat, summ_spec$Rhat)
all_bulk_ess <- c(summ_fixed$Bulk_ESS, summ_spec$Bulk_ESS)
all_tail_ess <- c(summ_fixed$Tail_ESS, summ_spec$Tail_ESS)

if (!is.null(summ_random)) {
    all_rhat <- c(all_rhat, summ_random$Rhat)
    all_bulk_ess <- c(all_bulk_ess, summ_random$Bulk_ESS)
    all_tail_ess <- c(all_tail_ess, summ_random$Tail_ESS)
}

cat("\n--- Convergence (all parameters) ---\n")
cat(sprintf("  Max Rhat:      %.4f %s\n", max(all_rhat, na.rm=TRUE), 
            ifelse(max(all_rhat, na.rm=TRUE) < 1.01, "[OK]", "[!] > 1.01")))
cat(sprintf("  Min Bulk ESS:  %.0f %s\n", min(all_bulk_ess, na.rm=TRUE),
            ifelse(min(all_bulk_ess, na.rm=TRUE) >= 400, "[OK]", "[!] < 400")))
cat(sprintf("  Min Tail ESS:  %.0f %s\n", min(all_tail_ess, na.rm=TRUE),
            ifelse(min(all_tail_ess, na.rm=TRUE) >= 400, "[OK]", "[!] < 400")))

cat("\n--- Fixed effects summary ---\n")
cat(sprintf("  Max Rhat:      %.4f\n", max(summ_fixed$Rhat)))
cat(sprintf("  Min Bulk ESS:  %.0f\n", min(summ_fixed$Bulk_ESS)))
cat(sprintf("  Min Tail ESS:  %.0f\n", min(summ_fixed$Tail_ESS)))

# Store for csv_results
summ <- summ_fixed
cat("======================================================================\n\n")

# ─── Extract posterior ─────────────────────────────────────────────
posterior <- as.data.frame(model)
S <- nrow(posterior)
N <- nrow(model_df)

coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")
cat("\nPosterior summaries:\n")
print(coef_df)

r2_vals <- as.numeric(brms::bayes_R2(model))
cat(sprintf("\nBayes R²: %.3f (95%% CI: %.3f - %.3f)\n",
            mean(r2_vals), quantile(r2_vals, 0.025), quantile(r2_vals, 0.975)))

# ─── Variance Decomposition ────────────────────────────────────────
eta_full <- t(brms::posterior_linpred(model, re_formula = NULL, transform = FALSE))
eta_no_re <- t(brms::posterior_linpred(model, re_formula = NA, transform = FALSE))

all_fixef_names <- rownames(brms::fixef(model))
post_colnames <- colnames(as.matrix(model))

fixef_cols <- sapply(all_fixef_names, function(fn) {
    candidate <- paste0("b_", fn)
    if (candidate %in% post_colnames) return(candidate)
    candidate_alt <- paste0("b_", gsub(":", ".", fn))
    if (candidate_alt %in% post_colnames) return(candidate_alt)
    stop(sprintf("Could not find posterior column for '%s'", fn))
})

fixef_samples <- as.matrix(model)[, fixef_cols, drop = FALSE]
colnames(fixef_samples) <- all_fixef_names

design_formula <- get_design_formula(int_level)
X <- model.matrix(design_formula, data = model_df)

X_colnames <- colnames(X)
fixef_reordered <- matrix(NA, nrow = S, ncol = ncol(X))
colnames(fixef_reordered) <- X_colnames

for (j in seq_along(X_colnames)) {
    xname <- X_colnames[j]
    if (xname %in% colnames(fixef_samples)) {
        fixef_reordered[, j] <- fixef_samples[, xname]
    } else if (xname == "(Intercept)" && "Intercept" %in% colnames(fixef_samples)) {
        fixef_reordered[, j] <- fixef_samples[, "Intercept"]
    } else {
        xname_alt <- gsub(":", ".", xname)
        if (xname_alt %in% colnames(fixef_samples)) {
            fixef_reordered[, j] <- fixef_samples[, xname_alt]
        } else {
            warning(sprintf("Could not find coefficient for '%s'", xname))
            fixef_reordered[, j] <- 0
        }
    }
}

eta_linear <- X %*% t(fixef_reordered)
eta_phylo <- eta_full - eta_no_re

y <- model_df$log_rate_median
SS_total <- sum((y - mean(y))^2)

intercept_samples <- fixef_reordered[, "(Intercept)"]
eta_linear_no_intercept <- eta_linear - matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE)

R2_full <- 1 - apply((matrix(y, nrow = N, ncol = S) - eta_full)^2, 2, sum) / SS_total

SS_resid_full <- apply((matrix(y, nrow = N, ncol = S) - eta_full)^2, 2, sum)

eta_linear_only <- matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE) + 
                   eta_linear_no_intercept
SS_resid_no_phylo_exact <- apply((matrix(y, nrow = N, ncol = S) - eta_linear_only)^2, 2, sum)

partial_R2_phylo <- (SS_resid_no_phylo_exact - SS_resid_full) / SS_total
prop_residual <- SS_resid_full / SS_total

# ─── Shapley Decomposition ─────────────────────────────────────────
term_names <- colnames(X)
terms_no_intercept <- setdiff(term_names, "(Intercept)")
p <- length(terms_no_intercept)

X_terms <- X[, terms_no_intercept, drop = FALSE]
X_terms_centered <- scale(X_terms, center = TRUE, scale = FALSE)

Q_list <- vector("list", 2^p)
Q_list[[1]] <- NULL

for (mask in 1:(2^p - 1)) {
    idx <- which(as.logical(intToBits(mask))[1:p])
    Q_list[[mask + 1]] <- qr.Q(qr(X_terms_centered[, idx, drop = FALSE]))
}

v <- vector("list", 2^p)
v[[1]] <- rep(0, S)

for (mask in 1:(2^p - 1)) {
    Q <- Q_list[[mask + 1]]
    Qt_mu <- crossprod(Q, eta_linear_no_intercept)
    mu_hat <- Q %*% Qt_mu
    v[[mask + 1]] <- apply(mu_hat^2, 2, sum)
}

fact <- factorial(0:p)
phi_SS <- matrix(0, nrow = S, ncol = p)
colnames(phi_SS) <- terms_no_intercept

for (i in seq_len(p)) {
    for (mask in 0:(2^p - 1)) {
        bits <- as.logical(intToBits(mask))[1:p]
        if (bits[i]) next
        
        s_size <- sum(bits)
        S_with_i <- mask + 2^(i - 1)
        
        weight <- fact[s_size + 1] * fact[p - s_size] / fact[p + 1]
        marginal <- v[[S_with_i + 1]] - v[[mask + 1]]
        phi_SS[, i] <- phi_SS[, i] + weight * marginal
    }
}

phi_R2 <- phi_SS / SS_total

term_partial_R2 <- list()
for (term in terms_no_intercept) {
    term_partial_R2[[term]] <- summarize_posterior(phi_R2[, term])
}

# ─── Normalize ─────────────────────────────────────────────────────
total_for_norm <- rowSums(pmax(phi_R2, 0)) + 
                  pmax(partial_R2_phylo, 0) + 
                  pmax(prop_residual, 0)

shapley_norm <- pmax(phi_R2, 0) / total_for_norm
prop_phylo_norm <- pmax(partial_R2_phylo, 0) / total_for_norm
prop_residual_norm <- pmax(prop_residual, 0) / total_for_norm

# ─── Save Posterior Samples ────────────────────────────────────────
samples_df <- data.frame(
    sample_id = 1:S,
    tree = tree_name,
    R2_full = R2_full,
    partial_R2_phylo = partial_R2_phylo,
    prop_residual = prop_residual,
    prop_phylo_norm = prop_phylo_norm,
    prop_residual_norm = prop_residual_norm
)

for (term in terms_no_intercept) {
    term_clean <- gsub(":", "_", term)
    samples_df[[paste0("shapley_", term_clean)]] <- phi_R2[, term]
    samples_df[[paste0("shapley_norm_", term_clean)]] <- shapley_norm[, term]
}

file_suffix <- paste0("linear_", int_level)

samples_path <- file.path(out_dir, 
    paste0("variance_samples_", file_suffix, "_", tree_name, "_", coord_method, ".csv"))
write.csv(samples_df, samples_path, row.names = FALSE)
cat("✔️ Posterior samples written to", samples_path, "\n")

coef_samples_df <- data.frame(
    sample_id = 1:S,
    tree = tree_name,
    fixef_samples
)
colnames(coef_samples_df) <- gsub("^b_", "", colnames(coef_samples_df))

coef_samples_path <- file.path(out_dir,
    paste0("coef_samples_", file_suffix, "_", tree_name, "_", coord_method, ".csv"))
write.csv(coef_samples_df, coef_samples_path, row.names = FALSE)
cat("✔️ Coefficient samples written to", coef_samples_path, "\n")

# ─── Save Summary Results ──────────────────────────────────────────
csv_results <- data.frame(
    model = "brms_phylo_linear",
    tree = tree_name,
    interaction_level = int_level,
    coordinate_method = coord_method,
    stringsAsFactors = FALSE
)

# Add MCMC diagnostics (ALL parameters + fixed effects separately)
csv_results$n_divergent <- n_div
csv_results$max_rhat <- max(all_rhat, na.rm = TRUE)
csv_results$min_bulk_ess <- min(all_bulk_ess, na.rm = TRUE)
csv_results$min_tail_ess <- min(all_tail_ess, na.rm = TRUE)
csv_results$max_rhat_fixed <- max(summ_fixed$Rhat, na.rm = TRUE)
csv_results$min_bulk_ess_fixed <- min(summ_fixed$Bulk_ESS, na.rm = TRUE)
csv_results$min_tail_ess_fixed <- min(summ_fixed$Tail_ESS, na.rm = TRUE)

for (k in seq_len(nrow(coef_df))) {
    term_name <- rownames(coef_df)[k]
    term_clean <- gsub(":", "_", term_name)
    csv_results[[paste0("coef_", term_clean)]] <- coef_df[k, "Estimate"]
    csv_results[[paste0("se_", term_clean)]] <- coef_df[k, "Est.Error"]
    csv_results[[paste0("ci_lower_", term_clean)]] <- coef_df[k, "CI_lower"]
    csv_results[[paste0("ci_upper_", term_clean)]] <- coef_df[k, "CI_upper"]
}

csv_results$R2_full_mean <- summarize_posterior(R2_full)$mean
csv_results$R2_full_q2_5 <- summarize_posterior(R2_full)$q2.5
csv_results$R2_full_q97_5 <- summarize_posterior(R2_full)$q97.5

csv_results$phylo_partial_R2_mean <- summarize_posterior(partial_R2_phylo)$mean
csv_results$phylo_partial_R2_q2_5 <- summarize_posterior(partial_R2_phylo)$q2.5
csv_results$phylo_partial_R2_q97_5 <- summarize_posterior(partial_R2_phylo)$q97.5

csv_results$residual_prop_mean <- summarize_posterior(prop_residual)$mean
csv_results$residual_prop_q2_5 <- summarize_posterior(prop_residual)$q2.5
csv_results$residual_prop_q97_5 <- summarize_posterior(prop_residual)$q97.5

for (term in terms_no_intercept) {
    term_clean <- gsub(":", "_", term)
    csv_results[[paste0("shapley_", term_clean, "_mean")]] <- term_partial_R2[[term]]$mean
    csv_results[[paste0("shapley_", term_clean, "_q2_5")]] <- term_partial_R2[[term]]$q2.5
    csv_results[[paste0("shapley_", term_clean, "_q97_5")]] <- term_partial_R2[[term]]$q97.5
}

csv_path <- file.path(out_dir,
    paste0("phylolm_", file_suffix, "_", tree_name, "_", coord_method, ".csv"))
write.csv(csv_results, csv_path, row.names = FALSE)
cat("✔️ CSV results written to", csv_path, "\n")

# ─── Print Summary ─────────────────────────────────────────────────
cat("\n")
cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
cat(sprintf("COMPLETED: %s × %s\n", tree_name, int_level))
cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
cat(sprintf("  R²: %.3f (95%% CI: %.3f – %.3f)\n",
            summarize_posterior(R2_full)$mean,
            summarize_posterior(R2_full)$q2.5,
            summarize_posterior(R2_full)$q97.5))
cat(sprintf("  Phylogenetic: %.1f%%\n", summarize_posterior(partial_R2_phylo)$mean * 100))
cat(sprintf("  Residual: %.1f%%\n", summarize_posterior(prop_residual)$mean * 100))
cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
