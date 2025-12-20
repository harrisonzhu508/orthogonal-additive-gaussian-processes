#!/usr/bin/env Rscript

# run_linear_interactions_phylo.R
#
# Bayesian phylogenetic regression with LINEAR lat/lon (no spline)
# Runs all interaction orders: main effects only, 2nd order, 3rd order, etc.
#
# Usage:
#   Rscript speech_phylo/final_scripts/run_linear_interactions_phylo.R

# ─── 0. Setup ──────────────────────────────────────────────────────
suppressPackageStartupMessages({
    library(ape)
    library(brms)
    library(dplyr)
    library(tibble)
})

# ensure output dir exists
out_dir <- "speech_phylo/final_phyloregression_results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ─── 1. Load BEAST helper & data ─────────────────────────────────
source("speech_phylo/beast.R")

tree_names <- c("long_v3_CCD0.0.25", "heggarty2024")
coord_methods <- c("standard")

# Define interaction levels to run
interaction_levels <- c(
    "main_only",      # Only main effects: lon, lat, log_n_speakers, n_segments, delta
    "secondorder",    # + 2nd order interactions (5 choose 2 = 10)
    "thirdorder",     # + 3rd order interactions (5 choose 3 = 10)
    "fourthorder",    # + 4th order interactions (5 choose 4 = 5)
    "fifthorder"      # + 5th order interaction (5 choose 5 = 1)
)

# Generate formulas for each interaction level
get_formula <- function(level) {
    base_terms <- "longitude_norm + latitude_norm + log_n_speakers_norm + n_segments_norm + delta_norm"
    
    # 2nd order interactions (5 choose 2 = 10)
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
    
    # 3rd order interactions (5 choose 3 = 10)
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
    
    # 4th order interactions (5 choose 4 = 5)
    fourth_order <- paste0(
        "longitude_norm:latitude_norm:log_n_speakers_norm:n_segments_norm + ",
        "longitude_norm:latitude_norm:log_n_speakers_norm:delta_norm + ",
        "longitude_norm:latitude_norm:n_segments_norm:delta_norm + ",
        "longitude_norm:log_n_speakers_norm:n_segments_norm:delta_norm + ",
        "latitude_norm:log_n_speakers_norm:n_segments_norm:delta_norm")
    
    # 5th order interaction (5 choose 5 = 1)
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

# Get design matrix formula for Shapley decomposition
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

# Helper function for summarizing posterior samples
summarize_posterior <- function(x) {
    list(
        mean  = mean(x, na.rm = TRUE),
        sd    = sd(x, na.rm = TRUE),
        q2.5  = unname(quantile(x, 0.025, na.rm = TRUE)),
        q50   = unname(quantile(x, 0.50, na.rm = TRUE)),
        q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
    )
}

# ═══════════════════════════════════════════════════════════════════════
# MAIN LOOP: Trees × Coord Methods × Interaction Levels
# ═══════════════════════════════════════════════════════════════════════

for (tree_name in tree_names) {
    cat(sprintf("\n%s\n", paste(rep("=", 70), collapse = "")))
    cat(sprintf("TREE: %s\n", tree_name))
    cat(sprintf("%s\n\n", paste(rep("=", 70), collapse = "")))
    
    tree_path <- paste0("speech_phylo/", tree_name, ".nex")
    tr <- read.annot.beast(tree_path)
    
    # Load precomputed regression data
    reg_data_path <- file.path(out_dir, paste0("metadata_", tree_name, "_with_inventory_delta.csv"))
    df <- read.csv(reg_data_path, stringsAsFactors = FALSE)
    rownames(df) <- df$language
    
    # Prune tree to match regression dataset
    tr <- drop.tip(tr, setdiff(tr$tip.label, rownames(df)))
    V_raw <- vcv(unroot(tr))
    V_raw <- V_raw[rownames(df), rownames(df)]
    
    # Normalize speaker count
    log_n_scaled <- scale(log(df$n_speakers))
    df$log_n_speakers_norm <- log_n_scaled[, 1]
    
    base_df <- df
    
    for (coord_method in coord_methods) {
        df_loop <- base_df
        
        # Normalize coordinates
        lon_stats <- apply_coord_scaling(df_loop$longitude, coord_method)
        lat_stats <- apply_coord_scaling(df_loop$latitude, coord_method)
        df_loop$longitude_norm <- lon_stats$values
        df_loop$latitude_norm <- lat_stats$values
        
        # Normalize n_segments and delta
        df_loop$n_segments_norm <- scale(df_loop$n_segments)[, 1]
        df_loop$delta_norm <- scale(df_loop$delta)[, 1]
        
        model_df <- df_loop
        model_df$log_rate_median <- log(model_df$rate_median)
        language_levels <- rownames(model_df)
        model_df$language_factor <- factor(language_levels, levels = language_levels)
        
        V <- V_raw[language_levels, language_levels]
        
        # ─── Loop through interaction levels ───────────────────────────────
        for (int_level in interaction_levels) {
            cat(sprintf("\n%s\n", paste(rep("-", 60), collapse = "")))
            cat(sprintf("Fitting: %s (%s)\n", int_level, coord_method))
            cat(sprintf("%s\n", paste(rep("-", 60), collapse = "")))
            
            formula_obj <- get_formula(int_level)
            
            # Set priors (no spline prior needed)
            priors <- c(
                set_prior("normal(0, 0.3)", class = "b"),
                set_prior("student_t(5, -6, 0.5)", class = "Intercept"),
                set_prior("exponential(2)", class = "sigma"),
                set_prior("exponential(2)", class = "sd", group = "language_factor")
            )
            
            brm_args <- list(
                formula = formula_obj,
                data = model_df,
                family = gaussian(),
                prior = priors,
                refresh = 0,
                chains = 4,
                iter = 6000,
                warmup = 2000,
                seed = 20231103,
                cores = max(1L, min(4L, parallel::detectCores())),
                control = list(adapt_delta = 0.99, max_treedepth = 15),
                data2 = list(V = V),
                sample_prior = "yes"
            )
            
            # Fit model
            model <- do.call(brm, brm_args)
            
            # Check diagnostics
            n_div <- sum(nuts_params(model, pars = "divergent__")$Value)
            cat(sprintf("Divergent transitions: %d (%.2f%%)\n", n_div, 100 * n_div / (4 * 4000)))
            
            summ <- summary(model)$fixed
            cat(sprintf("Max Rhat: %.3f | Min Bulk ESS: %.0f | Min Tail ESS: %.0f\n",
                        max(summ$Rhat), min(summ$Bulk_ESS), min(summ$Tail_ESS)))
            
            # Extract posterior
            posterior <- as.data.frame(model)
            S <- nrow(posterior)
            N <- nrow(model_df)
            
            # Coefficient summary
            coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
            colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")
            cat("\nPosterior summaries:\n")
            print(coef_df)
            
            # R² calculation
            r2_vals <- as.numeric(brms::bayes_R2(model))
            r2_summary <- list(
                mean = mean(r2_vals),
                sd = sd(r2_vals),
                q2.5 = quantile(r2_vals, 0.025),
                q97.5 = quantile(r2_vals, 0.975)
            )
            cat(sprintf("\nBayes R²: %.3f (95%% CI: %.3f - %.3f)\n",
                        r2_summary$mean, r2_summary$q2.5, r2_summary$q97.5))
            
            # ─── Variance Decomposition ─────────────────────────────────────
            # Extract components from posterior_linpred
            eta_full <- t(brms::posterior_linpred(model, re_formula = NULL, transform = FALSE))
            eta_no_re <- t(brms::posterior_linpred(model, re_formula = NA, transform = FALSE))
            
            # Get fixed effect samples
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
            
            # Build design matrix
            design_formula <- get_design_formula(int_level)
            X <- model.matrix(design_formula, data = model_df)
            
            # Match and reorder coefficients
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
            
            # Compute linear effects
            eta_linear <- X %*% t(fixef_reordered)
            
            # Phylogenetic effect
            eta_phylo <- eta_full - eta_no_re
            
            # Variance decomposition
            y <- model_df$log_rate_median
            SS_total <- sum((y - mean(y))^2)
            var_y <- var(y)
            
            # Remove intercept for variance calculation
            intercept_samples <- fixef_reordered[, "(Intercept)"]
            eta_linear_no_intercept <- eta_linear - matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE)
            
            # Compute R² for full model
            R2_full <- 1 - apply((matrix(y, nrow = N, ncol = S) - eta_full)^2, 2, sum) / SS_total
            
            # Compute partial R² for linear and phylogenetic components
            SS_resid_full <- apply((matrix(y, nrow = N, ncol = S) - eta_full)^2, 2, sum)
            SS_resid_no_phylo <- apply((matrix(y, nrow = N, ncol = S) - eta_no_re)^2, 2, sum)
            
            # Linear only fitted = intercept + linear
            eta_linear_only <- matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE) + 
                               eta_linear_no_intercept
            SS_resid_no_phylo_exact <- apply((matrix(y, nrow = N, ncol = S) - eta_linear_only)^2, 2, sum)
            
            partial_R2_phylo <- (SS_resid_no_phylo_exact - SS_resid_full) / SS_total
            partial_R2_linear <- apply(eta_linear_no_intercept^2, 2, sum) / SS_total
            
            # Residual proportion
            prop_residual <- SS_resid_full / SS_total
            
            # ─── Shapley Decomposition for Linear Terms ─────────────────────
            # Get term names (excluding intercept)
            term_names <- colnames(X)
            terms_no_intercept <- setdiff(term_names, "(Intercept)")
            p <- length(terms_no_intercept)
            
            X_terms <- X[, terms_no_intercept, drop = FALSE]
            X_terms_centered <- scale(X_terms, center = TRUE, scale = FALSE)
            
            # Precompute QR decompositions
            Q_list <- vector("list", 2^p)
            Q_list[[1]] <- NULL
            
            for (mask in 1:(2^p - 1)) {
                idx <- which(as.logical(intToBits(mask))[1:p])
                Q_list[[mask + 1]] <- qr.Q(qr(X_terms_centered[, idx, drop = FALSE]))
            }
            
            # Compute value function
            v <- vector("list", 2^p)
            v[[1]] <- rep(0, S)
            
            for (mask in 1:(2^p - 1)) {
                Q <- Q_list[[mask + 1]]
                Qt_mu <- crossprod(Q, eta_linear_no_intercept)
                mu_hat <- Q %*% Qt_mu
                v[[mask + 1]] <- apply(mu_hat^2, 2, sum)
            }
            
            # Compute Shapley values
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
            
            # Convert to R² scale
            phi_R2 <- phi_SS / SS_total
            
            # Summarize term contributions
            term_SS <- list()
            term_partial_R2 <- list()
            for (term in terms_no_intercept) {
                term_SS[[term]] <- summarize_posterior(phi_SS[, term])
                term_partial_R2[[term]] <- summarize_posterior(phi_R2[, term])
            }
            
            # ─── Normalize for Proportional Decomposition ───────────────────
            # Components: all Shapley terms + phylo + residual
            total_for_norm <- rowSums(pmax(phi_R2, 0)) + 
                              pmax(partial_R2_phylo, 0) + 
                              pmax(prop_residual, 0)
            
            # Normalize
            shapley_norm <- pmax(phi_R2, 0) / total_for_norm
            prop_phylo_norm <- pmax(partial_R2_phylo, 0) / total_for_norm
            prop_residual_norm <- pmax(prop_residual, 0) / total_for_norm
            
            # ─── Save Posterior Samples ─────────────────────────────────────
            samples_df <- data.frame(
                sample_id = 1:S,
                tree = tree_name,
                R2_full = R2_full,
                partial_R2_phylo = partial_R2_phylo,
                prop_residual = prop_residual,
                prop_phylo_norm = prop_phylo_norm,
                prop_residual_norm = prop_residual_norm
            )
            
            # Add Shapley columns
            for (term in terms_no_intercept) {
                term_clean <- gsub(":", "_", term)
                samples_df[[paste0("shapley_", term_clean)]] <- phi_R2[, term]
                samples_df[[paste0("shapley_norm_", term_clean)]] <- shapley_norm[, term]
            }
            
            # Save file suffix
            file_suffix <- paste0("linear_", int_level)
            
            samples_path <- file.path(out_dir, 
                paste0("variance_samples_", file_suffix, "_", tree_name, "_", coord_method, ".csv"))
            write.csv(samples_df, samples_path, row.names = FALSE)
            cat("✔️ Posterior samples written to", samples_path, "\n")
            
            # Save coefficient samples
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
            
            # ─── Save Summary Results ───────────────────────────────────────
            csv_results <- data.frame(
                model = "brms_phylo_linear",
                tree = tree_name,
                interaction_level = int_level,
                coordinate_method = coord_method,
                stringsAsFactors = FALSE
            )
            
            # Add coefficients
            for (i in seq_len(nrow(coef_df))) {
                term_name <- rownames(coef_df)[i]
                term_clean <- gsub(":", "_", term_name)
                csv_results[[paste0("coef_", term_clean)]] <- coef_df[i, "Estimate"]
                csv_results[[paste0("se_", term_clean)]] <- coef_df[i, "Est.Error"]
                csv_results[[paste0("ci_lower_", term_clean)]] <- coef_df[i, "CI_lower"]
                csv_results[[paste0("ci_upper_", term_clean)]] <- coef_df[i, "CI_upper"]
            }
            
            # Add variance decomposition
            csv_results$R2_full_mean <- summarize_posterior(R2_full)$mean
            csv_results$R2_full_q2_5 <- summarize_posterior(R2_full)$q2.5
            csv_results$R2_full_q97_5 <- summarize_posterior(R2_full)$q97.5
            
            csv_results$phylo_partial_R2_mean <- summarize_posterior(partial_R2_phylo)$mean
            csv_results$phylo_partial_R2_q2_5 <- summarize_posterior(partial_R2_phylo)$q2.5
            csv_results$phylo_partial_R2_q97_5 <- summarize_posterior(partial_R2_phylo)$q97.5
            
            csv_results$residual_prop_mean <- summarize_posterior(prop_residual)$mean
            csv_results$residual_prop_q2_5 <- summarize_posterior(prop_residual)$q2.5
            csv_results$residual_prop_q97_5 <- summarize_posterior(prop_residual)$q97.5
            
            # Add term Shapley values
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
            
            # Print summary
            cat("\n")
            cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
            cat(sprintf("VARIANCE DECOMPOSITION: %s\n", toupper(int_level)))
            cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
            cat(sprintf("  Model R²: %.3f (95%% CI: %.3f – %.3f)\n",
                        summarize_posterior(R2_full)$mean,
                        summarize_posterior(R2_full)$q2.5,
                        summarize_posterior(R2_full)$q97.5))
            cat(sprintf("  Phylogenetic: %.1f%%\n", summarize_posterior(partial_R2_phylo)$mean * 100))
            cat(sprintf("  Residual: %.1f%%\n", summarize_posterior(prop_residual)$mean * 100))
            cat("\n  Linear term Shapley values:\n")
            for (term in terms_no_intercept) {
                cat(sprintf("    %s: %.1f%%\n", term, term_partial_R2[[term]]$mean * 100))
            }
            cat(sprintf("══════════════════════════════════════════════════════════════════\n"))
        }
    }
}

cat("\n✅ All models completed!\n")
