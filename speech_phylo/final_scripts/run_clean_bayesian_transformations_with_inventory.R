#!/usr/bin/env Rscript

# export_results_to_json.R (FIXED VERSION)
#
# Usage:
#   Rscript speech_phylo/export_results_to_json.R

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
source("speech_phylo/beast.R")  # must define read.annot.beast() & extract_beast_metadata()

tree_names <- c("long_v3_CCD0.0.25", "heggarty2024")

coordinate_shift <- 10

apply_coord_scaling <- function(vec, method, shift = coordinate_shift) {
    forward_transform <- NULL
    description <- NULL
    shift_value <- NA_real_

    if (method == "standard") {
        forward_transform <- function(x) x
        description <- "scale(x)"
    } else if (method == "sqrt_plus_10") {
        if (any(vec + shift <= 0)) {
            stop(sprintf("sqrt_plus_10 requires all values to be greater than -%s", shift))
        }
        forward_transform <- function(x) sqrt(x + shift)
        description <- sprintf("scale(sqrt(x + %s))", shift)
        shift_value <- shift
    } else if (method == "log_plus_10") {
        if (any(vec + shift <= 0)) {
            stop(sprintf("log_plus_10 requires all values to be greater than -%s", shift))
        }
        forward_transform <- function(x) log(x + shift)
        description <- sprintf("scale(log(x + %s))", shift)
        shift_value <- shift
    } else {
        stop(sprintf("Unknown coordinate scaling method: %s", method))
    }

    transformed <- forward_transform(vec)
    scaled <- scale(transformed)
    center <- as.numeric(attr(scaled, "scaled:center"))
    scale_val <- as.numeric(attr(scaled, "scaled:scale"))

    return(list(
        values = as.numeric(scaled),
        center = center,
        scale = scale_val,
        description = description,
        shift = shift_value
    ))
}

# regression dataset is produced in Python:
#   df_meta_cognates.to_csv("final_phyloregression_results/metadata_heggarty2024_with_inventory_delta.csv", index=False)
#   df_meta_speech.to_csv("final_phyloregression_results/metadata_long_v3_CCD0.0.25_with_inventory_delta.csv", index=False)

for (tree_name in tree_names) {

    tree_path <- paste0("speech_phylo/", tree_name, ".nex")
    tr        <- read.annot.beast(tree_path)

    # Load precomputed regression data with delta + inventory columns
    reg_data_path <- file.path(
        "speech_phylo/final_phyloregression_results",
        paste0("metadata_", tree_name, "_with_inventory_delta.csv")
    )
    df <- read.csv(reg_data_path, stringsAsFactors = FALSE)
    rownames(df) <- df$language

    # Prune tree to match regression dataset (and order covariance accordingly)
    tr <- drop.tip(tr, setdiff(tr$tip.label, rownames(df)))
    V_raw <- vcv(unroot(tr))
    V_raw <- V_raw[rownames(df), rownames(df)]


    # ─── 7. Speaker count transform (fixed: log + scale) ─────────────
    log_n_scaled <- scale(log(df$n_speakers))
    df$log_n_speakers_norm <- log_n_scaled[, 1]
    log_n_center <- as.numeric(attr(log_n_scaled, "scaled:center"))
    log_n_scale  <- as.numeric(attr(log_n_scaled, "scaled:scale"))

    base_df <- df
    # coord_methods <- c("standard", "sqrt_plus_10", "log_plus_10")
    coord_methods <- c("standard")

    for (coord_method in coord_methods) {
        df_loop <- base_df

        lon_stats <- apply_coord_scaling(df_loop$longitude, coord_method)
        lat_stats <- apply_coord_scaling(df_loop$latitude, coord_method)

        df_loop$longitude_norm <- lon_stats$values
        df_loop$latitude_norm  <- lat_stats$values

        x_center <- lon_stats$center
        x_scale  <- lon_stats$scale
        y_center <- lat_stats$center
        y_scale  <- lat_stats$scale

        # normalize n_segments and delta
        n_segments_scaled <- scale(df_loop$n_segments)
        df_loop$n_segments_norm <- n_segments_scaled[, 1]
        delta_scaled <- scale(df_loop$delta)
        df_loop$delta_norm <- delta_scaled[, 1]

        model_df <- df_loop
        model_df$log_rate_median <- log(model_df$rate_median)
        language_levels <- rownames(model_df)
        model_df$language_factor <- factor(language_levels, levels = language_levels)

        V <- V_raw[language_levels, language_levels]

        # ─── FIXED: Use thin-plate spline instead of GP for stability ───
        # Splines are more numerically stable and capture similar spatial patterns
        # Reference: Wood (2017) "Generalized Additive Models"
        # OPTIMIZED: Reduced spline complexity for better stability
        formula_obj <- bf(
            log_rate_median ~
                t2(longitude_norm, latitude_norm, k = 5, bs = "tp") +
                log_n_speakers_norm + n_segments_norm + delta_norm +
                (1 | gr(language_factor, cov = V)),
            decomp = "QR"
        )

        # OPTIMIZED: More iterations, higher adapt_delta for better convergence
        brm_args <- list(
            formula = formula_obj,
            data = model_df,
            family = gaussian(),
            refresh = 0,
            chains = 4,
            iter = 30000,
            warmup = 10000,
            seed = 20231103,
            cores = max(1L, min(4L, parallel::detectCores())),
            control = list(adapt_delta = 0.9999, max_treedepth = 15),
            data2 = list(V = V),
            thin=10
        )

        ##### PRIORS AND MODEL FITTING #####
        # OPTIMIZED: Tighter, more informative priors for better regularization
        # priors <- c(
        #     set_prior("normal(0, 0.2)", class = "b"),
        #     set_prior("student_t(5, -6, 0.5)", class = "Intercept"),  # More informative for log-rate
        #     set_prior("exponential(2)", class = "sigma"),
        #     set_prior("exponential(2)", class = "sd", group = "language_factor"),
        #     # Spline smoothness parameter - tighter for stability
        #     set_prior("student_t(5, 0, 0.5)", class = "sds")
        # )
        # brm_args$prior <- priors
        # brm_args$sample_prior <- "yes"

        cat(sprintf("\nFitting phylogenetic spatial regression (%s)...\n", coord_method))
        model <- do.call(brm, brm_args)
        model_type <- "brms_phylo"
        
        # ─── Comprehensive Diagnostics ───────────────────────────────────
        cat("\n══════════════════════════════════════════════════════════════════\n")
        cat("                    MCMC DIAGNOSTICS\n")
        cat("══════════════════════════════════════════════════════════════════\n")
        
        # Divergent transitions
        np <- nuts_params(model)
        n_div <- sum(np[np$Parameter == "divergent__", "Value"])
        n_samples <- (brm_args$iter - brm_args$warmup) * brm_args$chains
        cat(sprintf("Divergent transitions: %d / %d (%.2f%%)\n", n_div, n_samples, 100 * n_div / n_samples))
        if (n_div > 0) {
            cat("  ⚠️  Divergences detected - consider increasing adapt_delta\n")
        }
        
        # Tree depth
        max_td <- sum(np[np$Parameter == "treedepth__", "Value"] >= brm_args$control$max_treedepth)
        cat(sprintf("Max treedepth reached: %d times (max=%d)\n", max_td, brm_args$control$max_treedepth))
        
        # BFMI (Bayesian Fraction of Missing Information)
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
                cat("  ⚠️  Low BFMI (<0.2) detected - may indicate poor posterior exploration\n")
            }
        }
        
        # Rhat and ESS for ALL parameters (fixed + random + sigma + sds)
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
        
        cat("\n─── Convergence (all parameters) ───\n")
        cat(sprintf("  Max Rhat:      %.4f %s\n", max(all_rhat, na.rm=TRUE), 
                    ifelse(max(all_rhat, na.rm=TRUE) < 1.01, "✓", "⚠️ > 1.01")))
        cat(sprintf("  Min Bulk ESS:  %.0f %s\n", min(all_bulk_ess, na.rm=TRUE),
                    ifelse(min(all_bulk_ess, na.rm=TRUE) >= 400, "✓", "⚠️ < 400")))
        cat(sprintf("  Min Tail ESS:  %.0f %s\n", min(all_tail_ess, na.rm=TRUE),
                    ifelse(min(all_tail_ess, na.rm=TRUE) >= 400, "✓", "⚠️ < 400")))
        
        cat("\n─── Fixed effects summary ───\n")
        cat(sprintf("  Max Rhat:      %.4f\n", max(summ_fixed$Rhat)))
        cat(sprintf("  Min Bulk ESS:  %.0f\n", min(summ_fixed$Bulk_ESS)))
        cat(sprintf("  Min Tail ESS:  %.0f\n", min(summ_fixed$Tail_ESS)))
        
        summ <- summ_fixed
        cat("══════════════════════════════════════════════════════════════════\n\n")
                    
        # Extract posterior samples
        posterior <- as.data.frame(model)

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

        # Extract variance components from posterior
        phylo_sd_samples <- posterior$sd_language_factor__Intercept
        sigma_samples <- posterior$sigma
        residual_var_samples <- sigma_samples^2

        # Phylogenetic variance: tr(V)/n * σ²_phylo
        V_mean_diag <- mean(diag(V))  # = tr(V)/n
        phylo_var_samples <- (phylo_sd_samples^2) * V_mean_diag

        coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
        colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")

        cat("\nPosterior summaries (normalized / transformed scale):\n")
        print(coef_df[, c("Estimate", "Est.Error", "CI_lower", "CI_upper")])

        # ─── R² and fitted values ─────────────────────────────────────
        r2_vals <- as.numeric(brms::bayes_R2(model))
        r_squared_summary <- list(
            mean  = mean(r2_vals),
            sd    = stats::sd(r2_vals),
            q2.5  = unname(stats::quantile(r2_vals, 0.025)),
            q97.5 = unname(stats::quantile(r2_vals, 0.975))
        )

        fitted_summary <- fitted(model, summary = TRUE, probs = c(0.025, 0.975))
        preds_df <- data.frame(
            language     = language_levels,
            fitted_mean  = fitted_summary[, "Estimate"],
            fitted_sd    = fitted_summary[, "Est.Error"],
            fitted_lower = fitted_summary[, "Q2.5"],
            fitted_upper = fitted_summary[, "Q97.5"],
            stringsAsFactors = FALSE
        )

        residuals <- model_df$log_rate_median - fitted_summary[, "Estimate"]
        rmse <- sqrt(mean(residuals^2, na.rm = TRUE))
        mae  <- mean(abs(residuals), na.rm = TRUE)

        # ─── Extract Spline Surface for Visualization ─────────────────────
        # Create a grid over the normalized longitude/latitude space
        cat("\nExtracting spline surface for visualization...\n")
        
        n_grid <- 200  # Grid resolution
        
        # Use fixed normalized bounds to cover Ireland-Bangladesh extent
        lon_seq <- seq(-3, 3, length.out = n_grid)
        lat_seq <- seq(-3, 3, length.out = n_grid)
        
        grid_df <- expand.grid(
            longitude_norm = lon_seq,
            latitude_norm = lat_seq
        )
        
        # Add other covariates at their mean values (for prediction)
        grid_df$log_n_speakers_norm <- 0  # mean of normalized variable
        grid_df$n_segments_norm <- 0
        grid_df$delta_norm <- 0
        
        # Create a dummy language factor (won't be used with re_formula = NA)
        grid_df$language_factor <- factor(
            rep(language_levels[1], nrow(grid_df)),
            levels = language_levels
        )
        
        # Predict spline effect only (no random effects)
        spline_pred <- fitted(
            model, 
            newdata = grid_df,
            re_formula = NA,  # Exclude random effects
            summary = TRUE,
            probs = c(0.025, 0.975)
        )
        
        # Also get posterior draws for probability calculation
        spline_draws <- fitted(
            model, 
            newdata = grid_df,
            re_formula = NA,
            summary = FALSE  # Get all posterior draws
        )
        
        # The prediction includes intercept + linear + spline
        # Subtract intercept and linear effects to get spline only
        intercept_mean <- mean(posterior$b_Intercept)
        intercept_draws <- posterior$b_Intercept  # Vector of intercept draws
        
        # Calculate proportion of posterior samples where spline > 0 at each location
        # spline_draws is n_draws x n_locations matrix
        spline_draws_centered <- sweep(spline_draws, 1, intercept_draws, "-")
        spline_prob_positive <- colMeans(spline_draws_centered > 0)
        
        # Spline effect = prediction - intercept (linear covariates are at 0)
        spline_surface <- data.frame(
            longitude_norm = grid_df$longitude_norm,
            latitude_norm = grid_df$latitude_norm,
            spline_mean = spline_pred[, "Estimate"] - intercept_mean,
            spline_lower = spline_pred[, "Q2.5"] - intercept_mean,
            spline_upper = spline_pred[, "Q97.5"] - intercept_mean,
            spline_prob_positive = spline_prob_positive
        )
        
        # Also save original coordinates for plotting
        # Reverse the normalization to get original lon/lat
        spline_surface$longitude <- (spline_surface$longitude_norm * x_scale) + x_center
        spline_surface$latitude <- (spline_surface$latitude_norm * y_scale) + y_center
        
        # Save spline surface to CSV
        spline_path <- file.path(
            "speech_phylo/final_phyloregression_results",
            paste0("spline_surface_", tree_name, "_", coord_method, ".csv")
        )
        write.csv(spline_surface, spline_path, row.names = FALSE)
        cat("✔️ Spline surface written to", spline_path, "\n")

        csv_results <- data.frame(
            model             = model_type,
            tree              = tree_name,
            coordinate_method = coord_method,
            stringsAsFactors  = FALSE
        )
        
        # Add MCMC diagnostics to csv_results (ALL parameters + fixed effects)
        csv_results$n_divergent <- n_div
        csv_results$max_rhat <- max(all_rhat, na.rm = TRUE)
        csv_results$min_bulk_ess <- min(all_bulk_ess, na.rm = TRUE)
        csv_results$min_tail_ess <- min(all_tail_ess, na.rm = TRUE)
        csv_results$max_rhat_fixed <- max(summ_fixed$Rhat, na.rm = TRUE)
        csv_results$min_bulk_ess_fixed <- min(summ_fixed$Bulk_ESS, na.rm = TRUE)
        csv_results$min_tail_ess_fixed <- min(summ_fixed$Tail_ESS, na.rm = TRUE)

        # ─── FIXED: Variance Decomposition via posterior_linpred ─────────────
        # 
        # Model structure:
        #   log_rate = Intercept + X @ β + f_spline(lon, lat) + u_phylo + ε
        #
        # We decompose variance using:
        #   - posterior_linpred(re_formula = NULL): full η (spline + phylo RE)
        #   - posterior_linpred(re_formula = NA): η without random effects (spline included)
        #   - X @ β: linear fixed effects only
        # ─────────────────────────────────────────────────────────────────────

        cat("\nExtracting variance components via posterior_linpred...\n")

        N <- nrow(model_df)
        S <- nrow(posterior)  # number of posterior samples

        # Full linear predictor: intercept + X@β + spline + phylo_RE
        eta_full <- t(brms::posterior_linpred(model, re_formula = NULL, transform = FALSE))  # N x S

        # Without random effects: intercept + X@β + spline
        eta_no_re <- t(brms::posterior_linpred(model, re_formula = NA, transform = FALSE))  # N x S

        # ─── Extract Fixed Effect Coefficients ───────────────────────────────
        # Get posterior samples of PARAMETRIC fixed effects only (not smooth terms)
        all_fixef_names <- rownames(brms::fixef(model))
        # Filter out smooth terms (t2, s, te, ti) - these have different naming in posterior
        fixef_names <- all_fixef_names[!grepl("^t2|^s\\(|^te\\(|^ti\\(", all_fixef_names)]
        
        # Get column names from posterior to find correct naming
        post_colnames <- colnames(as.matrix(model))
        
        # Build the coefficient column names - brms uses "b_" prefix for fixed effects
        # but Intercept is sometimes stored differently
        fixef_cols <- sapply(fixef_names, function(fn) {
            candidate <- paste0("b_", fn)
            if (candidate %in% post_colnames) {
                return(candidate)
            }
            # Try with underscores instead of colons for interactions
            candidate_alt <- paste0("b_", gsub(":", ".", fn))
            if (candidate_alt %in% post_colnames) {
                return(candidate_alt)
            }
            stop(sprintf("Could not find posterior column for '%s'. Available b_ cols: %s",
                         fn, paste(grep("^b_", post_colnames, value = TRUE), collapse = ", ")))
        })
        
        fixef_samples <- as.matrix(model)[, fixef_cols, drop = FALSE]
        colnames(fixef_samples) <- fixef_names

        # Build design matrix for linear fixed effects (excluding spline)
        # NOTE: This must match the parametric terms in the model formula
        X <- model.matrix(~ log_n_speakers_norm + n_segments_norm + delta_norm,
                          data = model_df)

        # Match coefficient names between X and fixef_samples
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
                # Try alternative naming conventions
                xname_alt <- gsub(":", ".", xname)
                if (xname_alt %in% colnames(fixef_samples)) {
                    fixef_reordered[, j] <- fixef_samples[, xname_alt]
                } else {
                    stop(sprintf("Could not find coefficient for '%s'. Available: %s",
                                 xname, paste(colnames(fixef_samples), collapse = ", ")))
                }
            }
        }

        # Compute linear fixed effects: X @ β^T (N x S)
        eta_linear <- X %*% t(fixef_reordered)  # N x S

        # ─── Isolate Spline/GP Effect ────────────────────────────────────────
        # spline_effect = η_no_re - η_linear
        gp_effect <- eta_no_re - eta_linear  # N x S

        # Sanity check
        gp_means <- colMeans(gp_effect)
        cat(sprintf("Spline mean across observations: mean=%.4f, sd=%.4f (should be ~0)\n",
                    mean(gp_means), sd(gp_means)))

        # ─── Isolate Phylogenetic Random Effect ──────────────────────────────
        # phylo_effect = η_full - η_no_re
        eta_phylo <- eta_full - eta_no_re  # N x S

        # ═══════════════════════════════════════════════════════════════════════
        # ANOVA-STYLE VARIANCE DECOMPOSITION (Type III Partial SS)
        # ═══════════════════════════════════════════════════════════════════════
        #
        # Theory: We decompose Var(y) into components using partial R² approach.
        # For each component, we compute the reduction in residual SS when that
        # component is added to a model containing all other components.
        #
        # This properly accounts for correlations between predictors.
        # Note: Sum of partial contributions may exceed total R² if components
        # are positively correlated (shared variance counted multiple times).
        # ───────────────────────────────────────────────────────────────────────
        
        cat("\n─── ANOVA-Style Variance Decomposition ───\n")
        
        # Response and its variance (constant across posterior draws)
        y <- model_df$log_rate_median
        SS_total <- sum((y - mean(y))^2)  # Total sum of squares
        var_y <- var(y)  # Total response variance
        
        cat(sprintf("Total response variance: Var(y) = %.4f\n", var_y))
        cat(sprintf("Total sum of squares: SS_total = %.4f (N = %d)\n", SS_total, N))
        
        # Remove intercept from linear predictor for variance calculation
        intercept_samples <- fixef_reordered[, "(Intercept)"]
        eta_linear_no_intercept <- eta_linear - matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE)
        
        # ─── Compute SS for each nested model configuration ───────────────────
        # Full model: fitted = intercept + linear + spline + phylo
        # We compute partial SS for each component by comparing:
        #   SS_partial(component) = SS_resid(without component) - SS_resid(full)
        
        # SS_residual for full model (per posterior draw)
        # Full fitted = eta_full (which includes intercept + linear + spline + phylo)
        resid_full <- sweep(matrix(y, nrow = N, ncol = S), 1, y, "-") + 
                      (matrix(y, nrow = N, ncol = S) - eta_full)
        # Simpler: residual = y - eta_full
        SS_resid_full <- apply((matrix(y, nrow = N, ncol = S) - eta_full)^2, 2, sum)
        
        # SS_residual without phylogenetic RE (intercept + linear + spline only)
        SS_resid_no_phylo <- apply((matrix(y, nrow = N, ncol = S) - eta_no_re)^2, 2, sum)
        
        # SS_residual without spline (intercept + linear + phylo only)
        eta_no_spline <- eta_linear + eta_phylo  # linear includes intercept
        SS_resid_no_spline <- apply((matrix(y, nrow = N, ncol = S) - eta_no_spline)^2, 2, sum)
        
        # SS_residual without linear fixed effects (intercept + spline + phylo)
        # eta_no_linear = intercept + spline + phylo
        eta_no_linear <- matrix(intercept_samples, nrow = N, ncol = S, byrow = TRUE) + 
                         gp_effect + eta_phylo
        SS_resid_no_linear <- apply((matrix(y, nrow = N, ncol = S) - eta_no_linear)^2, 2, sum)
        
        # ─── Type III Partial Sums of Squares ─────────────────────────────────
        # Each component's contribution = how much SS_resid increases when removed
        
        SS_partial_linear <- SS_resid_no_linear - SS_resid_full  # S-vector
        SS_partial_spline <- SS_resid_no_spline - SS_resid_full  # S-vector
        SS_partial_phylo <- SS_resid_no_phylo - SS_resid_full    # S-vector
        SS_residual <- SS_resid_full                              # S-vector
        
        # Partial R² values (proportion of total SS explained by each component)
        partial_R2_linear <- SS_partial_linear / SS_total
        partial_R2_spline <- SS_partial_spline / SS_total
        partial_R2_phylo <- SS_partial_phylo / SS_total
        prop_residual <- SS_residual / SS_total
        
        # Full model R²
        R2_full <- 1 - SS_resid_full / SS_total
        
        cat(sprintf("Model R²: mean = %.3f (95%% CI: %.3f – %.3f)\n",
                    mean(R2_full), quantile(R2_full, 0.025), quantile(R2_full, 0.975)))
        
        # Sanity check: partial contributions
        sum_partial <- partial_R2_linear + partial_R2_spline + partial_R2_phylo
        cat(sprintf("Sum of partial R² contributions: mean = %.3f\n", mean(sum_partial)))
        cat("  (May exceed total R² if components are correlated)\n")
        
        # ─── Shapley Values for Fixed Effects (within linear component) ───────
        # For decomposing the linear fixed effects into individual terms
        
        term_names <- colnames(X)
        terms_no_intercept <- setdiff(term_names, "(Intercept)")
        p <- length(terms_no_intercept)
        
        X_terms <- X[, terms_no_intercept, drop = FALSE]
        X_terms_centered <- scale(X_terms, center = TRUE, scale = FALSE)
        
        # Precompute QR decompositions for all 2^p subsets
        Q_list <- vector("list", 2^p)
        Q_list[[1]] <- NULL  # empty subset
        
        for (mask in 1:(2^p - 1)) {
            idx <- which(as.logical(intToBits(mask))[1:p])
            XS <- X_terms_centered[, idx, drop = FALSE]
            qrS <- qr(XS)
            Q_list[[mask + 1]] <- qr.Q(qrS)
        }
        
        # Value function v(S) = SS explained by subset S (for linear terms only)
        # We project eta_linear_no_intercept onto subsets and compute SS explained
        v <- vector("list", 2^p)
        v[[1]] <- rep(0, S)
        
        for (mask in 1:(2^p - 1)) {
            Q <- Q_list[[mask + 1]]
            Qt_mu <- crossprod(Q, eta_linear_no_intercept)
            mu_hat <- Q %*% Qt_mu  # N x S
            v[[mask + 1]] <- apply(mu_hat^2, 2, sum)  # SS explained by this subset
        }
        
        # Compute Shapley values for individual terms (SS scale)
        fact <- factorial(0:p)
        den <- fact[p + 1]
        
        phi_SS <- matrix(0, nrow = S, ncol = p)
        colnames(phi_SS) <- terms_no_intercept
        
        for (j in 1:p) {
            bitj <- bitwShiftL(1L, j - 1L)
            for (mask in 0:(2^p - 1)) {
                if (bitwAnd(mask, bitj) != 0L) next
                k <- sum(as.logical(intToBits(mask))[1:p])
                w <- (fact[k + 1] * fact[p - k]) / den
                phi_SS[, j] <- phi_SS[, j] + w * (v[[bitwOr(mask, bitj) + 1]] - v[[mask + 1]])
            }
        }
        
        # Convert Shapley SS to partial R² contributions
        phi_R2 <- phi_SS / SS_total
        
        # Verify Shapley efficiency (sum should equal total linear SS)
        SS_linear_total <- apply(eta_linear_no_intercept^2, 2, sum)
        shapley_sum <- rowSums(phi_SS)
        cat(sprintf("Shapley efficiency check: max|sum(φ) - SS_linear| = %.3e\n",
                    max(abs(shapley_sum - SS_linear_total))))
        
        # ─── Summarize Results ────────────────────────────────────────────────
        
        # Per-term contributions
        term_SS <- list()
        term_partial_R2 <- list()
        for (j in 1:p) {
            term_SS[[terms_no_intercept[j]]] <- summarize_posterior(phi_SS[, j])
            term_partial_R2[[terms_no_intercept[j]]] <- summarize_posterior(phi_R2[, j])
        }
        
        variance_decomposition <- list(
            # Method description
            method = "ANOVA_Type_III",
            
            # Total response variance (constant)
            total_response_variance = var_y,
            total_SS = SS_total,
            
            # Full model fit
            R2_full = summarize_posterior(R2_full),
            
            # Linear fixed effects (Type III partial)
            linear_partial_SS = summarize_posterior(SS_partial_linear),
            linear_partial_R2 = summarize_posterior(partial_R2_linear),
            
            # Spatial spline component (Type III partial)
            spline_partial_SS = summarize_posterior(SS_partial_spline),
            spline_partial_R2 = summarize_posterior(partial_R2_spline),
            
            # Phylogenetic random effect (Type III partial)
            phylo_partial_SS = summarize_posterior(SS_partial_phylo),
            phylo_partial_R2 = summarize_posterior(partial_R2_phylo),
            
            # Residual
            residual_SS = summarize_posterior(SS_residual),
            residual_prop = summarize_posterior(prop_residual),
            
            # Per-term breakdown (Shapley decomposition of linear component)
            term_SS = term_SS,
            term_partial_R2 = term_partial_R2,
            
            # Metadata
            V_trace_normalized = V_mean_diag,
            n_taxa = nrow(V),
            n_obs = N
        )
        
        # Also store samples for potential further analysis
        variance_samples <- list(
            SS_partial_linear = SS_partial_linear,
            SS_partial_spline = SS_partial_spline,
            SS_partial_phylo = SS_partial_phylo,
            SS_residual = SS_residual,
            R2_full = R2_full,
            phi_SS = phi_SS
        )
        
        # ─── Save posterior samples for violin plots ───────────────────────────
        # For proportional decomposition, normalize ALL components together
        # Components: Shapley(log_n_speakers, n_segments, delta) + spline + phylo + residual
        
        # Get Shapley values for individual linear terms (already in R² scale)
        shapley_log_n_speakers <- phi_R2[, "log_n_speakers_norm"]
        shapley_n_segments <- phi_R2[, "n_segments_norm"]
        shapley_delta <- phi_R2[, "delta_norm"]
        
        # Compute TOTAL for normalization: sum of all 6 components
        # Clip negative values to 0 before summing
        total_for_norm <- pmax(shapley_log_n_speakers, 0) + 
                          pmax(shapley_n_segments, 0) + 
                          pmax(shapley_delta, 0) + 
                          pmax(partial_R2_spline, 0) + 
                          pmax(partial_R2_phylo, 0) + 
                          pmax(prop_residual, 0)
        
        # Normalize each component so all 6 sum to 1 per sample
        shapley_norm_log_n_speakers <- pmax(shapley_log_n_speakers, 0) / total_for_norm
        shapley_norm_n_segments <- pmax(shapley_n_segments, 0) / total_for_norm
        shapley_norm_delta <- pmax(shapley_delta, 0) / total_for_norm
        prop_spline_norm <- pmax(partial_R2_spline, 0) / total_for_norm
        prop_phylo_norm <- pmax(partial_R2_phylo, 0) / total_for_norm
        prop_residual_norm <- pmax(prop_residual, 0) / total_for_norm
        
        # Also compute aggregated linear proportion (sum of 3 Shapley terms)
        prop_linear_norm <- shapley_norm_log_n_speakers + shapley_norm_n_segments + shapley_norm_delta
        
        # Create dataframe of posterior samples
        samples_df <- data.frame(
            sample_id = 1:S,
            tree = tree_name,
            R2_full = R2_full,
            # Raw (unnormalized) values - can exceed 100%
            partial_R2_linear = partial_R2_linear,
            partial_R2_spline = partial_R2_spline,
            partial_R2_phylo = partial_R2_phylo,
            prop_residual = prop_residual,
            shapley_log_n_speakers = shapley_log_n_speakers,
            shapley_n_segments = shapley_n_segments,
            shapley_delta = shapley_delta,
            # Normalized proportions (sum to 1 - use these for plotting!)
            prop_linear_norm = prop_linear_norm,
            prop_spline_norm = prop_spline_norm,
            prop_phylo_norm = prop_phylo_norm,
            prop_residual_norm = prop_residual_norm,
            shapley_norm_log_n_speakers = shapley_norm_log_n_speakers,
            shapley_norm_n_segments = shapley_norm_n_segments,
            shapley_norm_delta = shapley_norm_delta
        )
        
        samples_path <- file.path(
            "speech_phylo/final_phyloregression_results",
            paste0("variance_samples_", tree_name, "_", coord_method, ".csv")
        )
        write.csv(samples_df, samples_path, row.names = FALSE)
        cat("✔️ Posterior samples written to", samples_path, "\n")
        
        # ─── Save coefficient posterior samples for violin plots ───────────────
        coef_samples_df <- data.frame(
            sample_id = 1:S,
            tree = tree_name,
            fixef_samples
        )
        # Clean column names (remove b_ prefix if present)
        colnames(coef_samples_df) <- gsub("^b_", "", colnames(coef_samples_df))
        
        coef_samples_path <- file.path(
            "speech_phylo/final_phyloregression_results",
            paste0("coef_samples_", tree_name, "_", coord_method, ".csv")
        )
        write.csv(coef_samples_df, coef_samples_path, row.names = FALSE)
        cat("✔️ Coefficient posterior samples written to", coef_samples_path, "\n")

        cat("\n══════════════════════════════════════════════════════════════════\n")
        cat("ANOVA-STYLE VARIANCE DECOMPOSITION (Type III Partial SS)\n")
        cat("══════════════════════════════════════════════════════════════════\n")
        cat(sprintf("  Total Var(y):         %.4f\n", var_y))
        cat(sprintf("  Model R²:             %.3f (95%% CI: %.3f – %.3f)\n",
                    variance_decomposition$R2_full$mean,
                    variance_decomposition$R2_full$q2.5,
                    variance_decomposition$R2_full$q97.5))
        cat("\n  Partial R² (Type III - controlling for all other components):\n")
        cat(sprintf("    Linear fixed effects: SS=%.4f (partial R²=%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
                    variance_decomposition$linear_partial_SS$mean,
                    variance_decomposition$linear_partial_R2$mean * 100,
                    variance_decomposition$linear_partial_R2$q2.5 * 100,
                    variance_decomposition$linear_partial_R2$q97.5 * 100))
        cat(sprintf("    Spatial (spline):     SS=%.4f (partial R²=%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
                    variance_decomposition$spline_partial_SS$mean,
                    variance_decomposition$spline_partial_R2$mean * 100,
                    variance_decomposition$spline_partial_R2$q2.5 * 100,
                    variance_decomposition$spline_partial_R2$q97.5 * 100))
        cat(sprintf("    Phylogenetic RE:      SS=%.4f (partial R²=%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
                    variance_decomposition$phylo_partial_SS$mean,
                    variance_decomposition$phylo_partial_R2$mean * 100,
                    variance_decomposition$phylo_partial_R2$q2.5 * 100,
                    variance_decomposition$phylo_partial_R2$q97.5 * 100))
        cat(sprintf("    Residual:             SS=%.4f (prop=%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
                    variance_decomposition$residual_SS$mean,
                    variance_decomposition$residual_prop$mean * 100,
                    variance_decomposition$residual_prop$q2.5 * 100,
                    variance_decomposition$residual_prop$q97.5 * 100))
        cat("\n  Per-term (Shapley decomposition of linear component):\n")
        for (term in names(term_partial_R2)) {
            cat(sprintf("    %-40s: SS=%.4f (partial R²=%.1f%%)\n", term,
                        term_SS[[term]]$mean,
                        term_partial_R2[[term]]$mean * 100))
        }
        cat("══════════════════════════════════════════════════════════════════\n")

        # Add coefficient information (normalized / transformed scale)
        for (i in seq_len(nrow(coef_df))) {
            term_name       <- rownames(coef_df)[i]
            term_name_clean <- gsub("_norm", "", term_name)  # Remove _norm suffix
            term_name_clean <- gsub(":", "_", term_name_clean)  # Replace : with _

            csv_results[paste0("coef_", term_name_clean)]      <- coef_df[i, "Estimate"]
            csv_results[paste0("se_", term_name_clean)]        <- coef_df[i, "Est.Error"]
            csv_results[paste0("ci_lower_", term_name_clean)]  <- coef_df[i, "CI_lower"]
            csv_results[paste0("ci_upper_", term_name_clean)]  <- coef_df[i, "CI_upper"]
        }

        ## ANOVA-style Variance decomposition (Type III partial SS)
        csv_results$method <- variance_decomposition$method
        csv_results$total_response_variance <- variance_decomposition$total_response_variance
        csv_results$total_SS <- variance_decomposition$total_SS
        
        # Full model R²
        csv_results$R2_full_mean <- variance_decomposition$R2_full$mean
        csv_results$R2_full_q2_5 <- variance_decomposition$R2_full$q2.5
        csv_results$R2_full_q97_5 <- variance_decomposition$R2_full$q97.5
        
        # Per-term SS and partial R² (Shapley decomposition of linear component)
        for (term in names(term_SS)) {
            term_clean <- gsub(":", "_", term)
            term_clean <- gsub("\\(|\\)", "", term_clean)
            csv_results[[paste0("SS_", term_clean)]] <- term_SS[[term]]$mean
            csv_results[[paste0("partial_R2_", term_clean)]] <- term_partial_R2[[term]]$mean
            csv_results[[paste0("partial_R2_", term_clean, "_q2_5")]] <- term_partial_R2[[term]]$q2.5
            csv_results[[paste0("partial_R2_", term_clean, "_q97_5")]] <- term_partial_R2[[term]]$q97.5
        }
        
        # Linear fixed effects (Type III)
        csv_results$linear_partial_SS_mean <- variance_decomposition$linear_partial_SS$mean
        csv_results$linear_partial_R2_mean <- variance_decomposition$linear_partial_R2$mean
        csv_results$linear_partial_R2_q2_5 <- variance_decomposition$linear_partial_R2$q2.5
        csv_results$linear_partial_R2_q97_5 <- variance_decomposition$linear_partial_R2$q97.5

        # Spline spatial component (Type III)
        csv_results$spline_partial_SS_mean <- variance_decomposition$spline_partial_SS$mean
        csv_results$spline_partial_R2_mean <- variance_decomposition$spline_partial_R2$mean
        csv_results$spline_partial_R2_q2_5 <- variance_decomposition$spline_partial_R2$q2.5
        csv_results$spline_partial_R2_q97_5 <- variance_decomposition$spline_partial_R2$q97.5

        # Phylogenetic random effect (Type III)
        csv_results$phylo_partial_SS_mean <- variance_decomposition$phylo_partial_SS$mean
        csv_results$phylo_partial_R2_mean <- variance_decomposition$phylo_partial_R2$mean
        csv_results$phylo_partial_R2_q2_5 <- variance_decomposition$phylo_partial_R2$q2.5
        csv_results$phylo_partial_R2_q97_5 <- variance_decomposition$phylo_partial_R2$q97.5
        
        # Residual
        csv_results$residual_SS_mean <- variance_decomposition$residual_SS$mean
        csv_results$residual_prop_mean <- variance_decomposition$residual_prop$mean
        csv_results$residual_prop_q2_5 <- variance_decomposition$residual_prop$q2.5
        csv_results$residual_prop_q97_5 <- variance_decomposition$residual_prop$q97.5
        
        # Metadata
        csv_results$V_trace_norm <- variance_decomposition$V_trace_normalized
        csv_results$n_obs <- variance_decomposition$n_obs

        # Save CSV
        csv_path <- file.path(
            "speech_phylo/final_phyloregression_results",
            paste0("phylolm_with_inventory_", tree_name, "_", coord_method, ".csv")
        )

        write.csv(csv_results, csv_path, row.names = FALSE)
        cat("✔️ CSV results written to", csv_path, "\n")
    }
}