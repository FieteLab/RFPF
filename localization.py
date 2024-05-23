import numpy as np
import matplotlib.pyplot as plt

# Increase font size
plt.rcParams.update({'font.size': 16})


# Monte Carlo localization
def mcl(epsilon=0.1):
    def run_mcl(particles, x, u):
        # Move particles randomly
        particles += epsilon * np.random.randn(*particles.shape)  # Shape (n, m)

        # Compute the loss of each particle
        normalized_particles = particles - u  # Shape (n, m)
        loss = 0.5 * np.sum(normalized_particles * x, axis=1) ** 2  # Shape (n,)
        likelihood = np.exp(-loss)  # Shape (n,)
        likelihood /= np.sum(likelihood)  # Shape (n,)

        # Resample particles
        indices = np.random.choice(len(particles), len(particles), p=likelihood)  # Shape (n,)
        particles = particles[indices]  # Shape (n, m)

        return particles

    return run_mcl


def C(m):
    # Gamma function
    return np.math.gamma(m / 2 + 1) / (m * (m - 2) * np.pi ** (m / 2))


# Our localization algorithm
def ours(gamma=0.1):
    def run_ours(particles, x, u):
        n = particles.shape[0]
        m = particles.shape[1]

        # Compute the loss of each particle
        normalized_particles = particles - u  # Shape (n, m)
        loss = 0.5 * np.sum(normalized_particles * x, axis=1) ** 2  # Shape (n,)

        # Normalized loss
        loss_normed = loss - np.mean(loss)  # Shape (n,)

        # Gradient of loss with respect to particles
        grad = np.outer(np.sum(normalized_particles * x, axis=1), x)  # Shape (n, m)

        # Distances
        deltas = particles[:, None, :] - particles[None, :, :]  # Shape (n, n, m)
        distances = np.linalg.norm(deltas, axis=2)  # Shape (n, n)

        epsilon = (gamma / C(m)) ** (1 / (2 - m))
        weights = (distances + epsilon) ** (-m)

        global_update = np.sum(deltas * weights[:, :, None] * loss_normed[:, None, None], axis=1)  # Shape (n, m)

        # Update particles
        particles -= gamma * grad / n + global_update * C(m) * (m - 2) / n
        return particles

    return run_ours


def KL_div(particles, u, t):
    est_mean = np.mean(particles, axis=0)  # Shape (m,)
    est_cov = np.cov(particles, rowvar=False)  # Shape (m, m)
    # Add small identity to avoid division by 0
    est_cov += 1e-6 * np.eye(est_cov.shape[0])
    m = est_mean.shape[0]

    # The other Gaussian has mean u and covariance I
    KL = 0.5 * (-m * np.log(t + 1) - np.log(np.linalg.det(est_cov)) - m + np.trace(est_cov) * (t + 1)
                + np.dot(est_mean - u * (t / (t + 1)), est_mean - u * (t / (t + 1))) * (t + 1))
    return KL


def test(update_fn, m=3, n=10000, iters=50):
    # Random vector with m dimensions
    u = np.random.randn(m)  # Shape (m,)

    # Sample particles
    particles = np.random.randn(n, m)  # Shape (n, m)

    discs = []
    for i in range(iters):
        discrepancy = KL_div(particles, u, i)

        x = np.random.randn(m)  # Shape (m,)
        particles = update_fn(particles, x, u)
        discs.append(discrepancy)
    return discs


if __name__ == '__main__':
    trials = 10
    epsilon = 0.1
    iters = 50
    ns = [20, 50, 100]

    best_epsilon = 0.1
    best_gamma = 0.1


    # Plot 1: KL divergence over iterations
    m = 10
    colors = ['xkcd:light blue', 'xkcd:blue', 'xkcd:dark blue']
    ls = [':', '--', '-']

    for n, color, l in zip(ns, colors, ls):
        discss = []
        for i in range(trials):
            print('n={}, trial={}'.format(n, i))
            discs = test(mcl(epsilon=best_epsilon), m=m, n=n, iters=iters)
            discss.append(np.asarray(discs))
        discss = np.asarray(discss)

        # Compute mean and standard error
        mean_discs = np.mean(np.asarray(discss), axis=0)
        std_discs = np.std(np.asarray(discss), axis=0) / np.sqrt(trials)

        # Plot with margins
        plt.plot(mean_discs, label='MCL, n={}'.format(n), color=color, linestyle=l)
        plt.fill_between(range(len(discs)), mean_discs - std_discs, mean_discs + std_discs, alpha=0.2,
                         color=color)

    colors = ['xkcd:light red', 'xkcd:red', 'xkcd:dark red']
    ls = [':', '--', '-']

    for n, color, l in zip(ns, colors, ls):
        discss = []
        for i in range(trials):
            print('n={}, trial={}'.format(n, i))
            discs = test(ours(gamma=best_gamma * n), m=m, n=n, iters=iters)
            discss.append(np.asarray(discs))
        discss = np.asarray(discss)

        # Compute mean and standard error
        mean_discs = np.mean(np.asarray(discss), axis=0)
        std_discs = np.std(np.asarray(discss), axis=0) / np.sqrt(trials)

        # Plot with margins
        plt.plot(mean_discs, label='Ours, n={}'.format(n), color=color, linestyle=l)
        plt.fill_between(range(len(discs)), mean_discs - std_discs, mean_discs + std_discs, alpha=0.2,
                         color=color)

    plt.xlabel('Iteration')
    plt.ylabel('KL Divergence with Posterior')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Plot 2: KL divergence over m
    ms = range(3, 11)
    colors = ['xkcd:light blue', 'xkcd:blue', 'xkcd:dark blue']
    ls = [':', '--', '-']

    for n, color, l in zip(ns, colors, ls):
        discss = []
        for i in range(trials):
            discs = []
            for m in ms:
                print('n={}, m={}, trial={}'.format(n, m, i))
                disc = test(mcl(epsilon=best_epsilon), m=m, n=n, iters=iters)
                discs.append(disc[-1])
            discss.append(np.asarray(discs))
        discss = np.asarray(discss)

        # Compute mean and standard error
        mean_discs = np.mean(np.asarray(discss), axis=0)
        std_discs = np.std(np.asarray(discss), axis=0) / np.sqrt(trials)

        # Plot with margins
        plt.plot(ms, mean_discs, label='MCL, n={}'.format(n), color=color, linestyle=l)
        plt.fill_between(ms, mean_discs - std_discs, mean_discs + std_discs, alpha=0.2,
                         color=color)

    colors = ['xkcd:light red', 'xkcd:red', 'xkcd:dark red']
    ls = [':', '--', '-']

    for n, color, l in zip(ns, colors, ls):
        discss = []
        for i in range(trials):
            discs = []
            for m in ms:
                print('n={}, m={}, trial={}'.format(n, m, i))
                disc = test(ours(gamma=best_gamma * n), m=m, n=n, iters=iters)
                discs.append(disc[-1])
            discss.append(np.asarray(discs))
        discss = np.asarray(discss)

        # Compute mean and standard error
        mean_discs = np.mean(np.asarray(discss), axis=0)
        std_discs = np.std(np.asarray(discss), axis=0) / np.sqrt(trials)

        # Plot with margins
        plt.plot(ms, mean_discs, label='Ours, n={}'.format(n), color=color, linestyle=l)
        plt.fill_between(ms, mean_discs - std_discs, mean_discs + std_discs, alpha=0.2,
                         color=color)

    # Log scale y axis
    plt.yscale('log')
    plt.xlabel('Dimensions')
    plt.ylabel('KL Divergence with Posterior')
    plt.tight_layout()
    plt.legend()
    plt.show()
